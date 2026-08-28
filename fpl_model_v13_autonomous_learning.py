import pandas as pd
import numpy as np
import requests

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import milp, LinearConstraint, Bounds, minimize
from scipy.sparse import lil_matrix


# ============================================================
# CONFIGURATION
# ============================================================

FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
FPL_ELEMENT_SUMMARY_URL = "https://fantasy.premierleague.com/api/element-summary/{player_id}/"

OUTPUT_FILE = "fpl_model_output_v13_autonomous_learning.xlsx"

UPCOMING_FIXTURE_COUNT = 3
N_CLUSTERS = 3
RANDOM_STATE = 42

# Long-term player quality.
# Price/value is deliberately excluded from player quality because
# price is already enforced by the Wildcard budget constraint.
RETENTION_WEIGHTS = {
    "total_points": 0.30,
    "points_per_game": 0.20,
    "minutes_security": 0.15,
    "form": 0.10,
    "ict_index": 0.10,
    "expected_goal_involvements": 0.10,
    "bonus": 0.05,
}

# Captaincy ceiling is separate from general retention quality.
CAPTAIN_WEIGHTS = {
    "points_per_game": 0.40,
    "total_points": 0.20,
    "xgi_per_90": 0.25,
    "ict_index": 0.10,
    "minutes_security": 0.05,
}

# V6 position-aware scoring.
# Price remains outside player quality and is handled only by the optimizer.
POSITION_SCORE_WEIGHTS = {
    "goalkeeper": {
        "retention": 0.60,
        "team_context": 0.20,
        "fixtures": 0.20,
    },
    "defender": {
        "retention": 0.60,
        "team_context": 0.20,
        "fixtures": 0.20,
    },
    "midfielder": {
        "retention": 0.60,
        "team_context": 0.20,
        "fixtures": 0.20,
    },
    "forward": {
        "retention": 0.60,
        "team_context": 0.20,
        "fixtures": 0.20,
    },
}

# Captaincy is based on the SINGLE next fixture, not the 3-fixture horizon.
# The next fixture is the dominant factor; individual ceiling breaks ties
# between strong captain options.
CAPTAIN_SINGLE_FIXTURE_WEIGHTS = {
    "next_fixture": 0.60,
    "individual_ceiling": 0.25,
    "team_attack": 0.10,
    "minutes_security": 0.05,
}

# Nearer fixtures matter more. With UPCOMING_FIXTURE_COUNT = 3 this gives
# GW+1 50%, GW+2 30%, GW+3 20%.
FIXTURE_RECENCY_WEIGHTS = [0.50, 0.30, 0.20]

TOP_COUNTS = {
    "goalkeeper": 15,
    "defender": 15,
    "midfielder": 15,
    "forward": 20,
}

# Full Wildcard rules.
WILDCARD_BUDGET = 1000  # £100.0m, because now_cost is in tenths.
MAX_PLAYERS_PER_CLUB = 3

SQUAD_POSITION_COUNTS = {
    "goalkeeper": 2,
    "defender": 5,
    "midfielder": 5,
    "forward": 3,
}

STARTER_WEIGHT = 1.00

# Bench philosophy:
# Bench players receive no model-score reward.
# The optimizer first maximizes Starting XI + Captain quality,
# then minimizes the total price of the four substitutes.
BENCH_OPTIMALITY_TOLERANCE = 1e-6


# ============================================================
# GAMEWEEK WEIGHT LEARNING
# ============================================================

WEIGHT_HISTORY_FILE = "fpl_gameweek_weight_history.csv"
NEXT_WEIGHT_ESTIMATOR = "median"
MIN_PLAYERS_FOR_WEIGHT_FIT = 8


# Proper learning workflow:
#   1. BEFORE a GW deadline, save the model feature snapshot.
#   2. AFTER that GW finishes, join actual GW points to THAT saved snapshot.
#   3. Fit and persist the learned weights.
#
# This prevents post-match data leakage.
AUTO_SAVE_NEXT_GW_SNAPSHOT = True
AUTO_LEARN_FROM_SAVED_SNAPSHOTS = True

SNAPSHOT_DIRECTORY = "fpl_snapshots"

# Autonomous-run state.
# This lets a scheduled process run repeatedly without duplicating work.
AUTOMATION_STATE_FILE = "fpl_automation_state.csv"

# We want the freshest possible PRE-GW snapshot, but we must never write one
# after the Gameweek deadline. A scheduled runner should execute this script
# regularly (for example hourly).

POSITION_LEARNING_FEATURES = {
    "goalkeeper": [
        "points_per_game_scaled",
        "minutes_security_scaled",
        "form_scaled",
        "ict_index_scaled",
        "bonus_scaled",
        "team_defence_score",
        "fixture_score",
    ],
    "defender": [
        "points_per_game_scaled",
        "minutes_security_scaled",
        "form_scaled",
        "ict_index_scaled",
        "expected_goal_involvements_scaled",
        "xgi_per_90_scaled",
        "bonus_scaled",
        "team_defence_score",
        "fixture_score",
    ],
    "midfielder": [
        "points_per_game_scaled",
        "minutes_security_scaled",
        "form_scaled",
        "ict_index_scaled",
        "expected_goal_involvements_scaled",
        "xgi_per_90_scaled",
        "bonus_scaled",
        "team_attack_score",
        "fixture_score",
    ],
    "forward": [
        "points_per_game_scaled",
        "minutes_security_scaled",
        "form_scaled",
        "ict_index_scaled",
        "expected_goal_involvements_scaled",
        "xgi_per_90_scaled",
        "bonus_scaled",
        "team_attack_score",
        "fixture_score",
    ],
}

CAPTAIN_LEARNING_FEATURES = [
    "next_fixture_score",
    "individual_captain_ceiling",
    "team_attack_score",
    "minutes_security",
]


# ============================================================
# DATA FETCHING
# ============================================================

def get_json(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def get_fpl_data():
    bootstrap = get_json(FPL_BOOTSTRAP_URL)
    fixtures = get_json(FPL_FIXTURES_URL)

    players_df = pd.DataFrame(bootstrap["elements"])
    teams_df = pd.DataFrame(bootstrap["teams"])
    fixtures_df = pd.DataFrame(fixtures)
    events_df = get_bootstrap_events()

    return players_df, teams_df, fixtures_df


# ============================================================
# HELPERS
# ============================================================

def numeric_series(df, column, default=0.0):
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)

    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).fillna(default)


def safe_divide(numerator, denominator):
    numerator = pd.to_numeric(
        numerator,
        errors="coerce",
    ).fillna(0.0)

    denominator = pd.to_numeric(
        denominator,
        errors="coerce",
    ).fillna(0.0)

    return np.where(
        denominator > 0,
        numerator / denominator,
        0.0,
    )


def minmax_scale_series(series):
    series = pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0.0)

    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        return pd.Series(
            0.0,
            index=series.index,
        )

    return (
        series - min_value
    ) / (
        max_value - min_value
    )


def weighted_score(players, weight_map):
    """
    Build a 0..1 weighted score.
    Zero-variance features are ignored and remaining weights are
    automatically renormalized. This is important in preseason
    when fields such as Form may be zero for every player.
    """
    active = {}

    for component_name, weight in weight_map.items():
        column = f"{component_name}_scaled"

        if column not in players.columns:
            continue

        values = players[column]

        if values.max() > values.min():
            active[component_name] = weight

    if not active:
        return pd.Series(
            0.0,
            index=players.index,
        )

    total_weight = sum(active.values())

    score = pd.Series(
        0.0,
        index=players.index,
    )

    for component_name, weight in active.items():
        score += (
            players[f"{component_name}_scaled"]
            * (weight / total_weight)
        )

    return score


# ============================================================
# PLAYER PREPARATION
# ============================================================

def prepare_players(players_df, teams_df):
    players = players_df.copy()

    team_map = (
        teams_df
        .set_index("id")["name"]
        .to_dict()
    )

    players["team_name"] = players["team"].map(
        team_map
    )

    position_mapping = {
        1: "goalkeeper",
        2: "defender",
        3: "midfielder",
        4: "forward",
    }

    players["position"] = players[
        "element_type"
    ].map(position_mapping)

    # Remove clearly unavailable players.
    if "status" in players.columns:
        players = players[
            ~players["status"].isin(
                ["i", "u", "s"]
            )
        ].copy()

    players["total_points_num"] = numeric_series(
        players,
        "total_points",
    )

    players["minutes_num"] = numeric_series(
        players,
        "minutes",
    )

    players["now_cost_num"] = numeric_series(
        players,
        "now_cost",
    )

    players["form_num"] = numeric_series(
        players,
        "form",
    )

    players["points_per_game_num"] = numeric_series(
        players,
        "points_per_game",
    )

    players["ict_index_num"] = numeric_series(
        players,
        "ict_index",
    )

    players["bonus_num"] = numeric_series(
        players,
        "bonus",
    )

    players["expected_goals_num"] = numeric_series(
        players,
        "expected_goals",
    )

    players["expected_assists_num"] = numeric_series(
        players,
        "expected_assists",
    )

    players["price"] = (
        players["now_cost_num"] / 10.0
    )

    # Informational only. It is NOT used in player quality.
    players["value"] = safe_divide(
        players["total_points_num"],
        players["price"],
    )

    if "expected_goal_involvements" in players.columns:
        players[
            "expected_goal_involvements_num"
        ] = numeric_series(
            players,
            "expected_goal_involvements",
        )
    else:
        players[
            "expected_goal_involvements_num"
        ] = (
            players["expected_goals_num"]
            + players["expected_assists_num"]
        )

    players["xgi_per_90"] = np.where(
        players["minutes_num"] > 0,
        (
            players[
                "expected_goal_involvements_num"
            ]
            / players["minutes_num"]
            * 90.0
        ),
        0.0,
    )

    max_minutes = players[
        "minutes_num"
    ].max()

    if max_minutes > 0:
        players["minutes_security"] = (
            players["minutes_num"]
            / max_minutes
        ).clip(0, 1)
    else:
        players["minutes_security"] = 0.0

    return players



# ============================================================
# CURRENT CLUB CONTEXT
# ============================================================

def add_team_context(players, teams_df):
    """
    Add current-club attack and defence strength from the live FPL teams table.

    This is deliberately separate from historical player production:
      - GK/DEF benefit from current club defensive strength.
      - MID/FWD benefit from current club attacking strength.
      - Captaincy uses current club attacking strength.

    Home and away team-strength values are averaged so the score describes
    the club itself; the fixture component separately handles the next games.
    """
    teams = teams_df.copy()

    required = [
        "id",
        "strength_attack_home",
        "strength_attack_away",
        "strength_defence_home",
        "strength_defence_away",
    ]

    missing = [column for column in required if column not in teams.columns]
    if missing:
        raise RuntimeError(
            "The FPL teams endpoint is missing required strength fields: "
            + ", ".join(missing)
        )

    for column in required[1:]:
        teams[column] = pd.to_numeric(
            teams[column],
            errors="coerce",
        )

    teams["team_attack_raw"] = teams[
        ["strength_attack_home", "strength_attack_away"]
    ].mean(axis=1)

    teams["team_defence_raw"] = teams[
        ["strength_defence_home", "strength_defence_away"]
    ].mean(axis=1)

    teams["team_attack_score"] = minmax_scale_series(
        teams["team_attack_raw"]
    )

    teams["team_defence_score"] = minmax_scale_series(
        teams["team_defence_raw"]
    )

    context = teams[
        [
            "id",
            "team_attack_raw",
            "team_defence_raw",
            "team_attack_score",
            "team_defence_score",
        ]
    ].rename(columns={"id": "team"})

    players = players.merge(
        context,
        how="left",
        on="team",
    )

    players["team_attack_score"] = players[
        "team_attack_score"
    ].fillna(0.5)

    players["team_defence_score"] = players[
        "team_defence_score"
    ].fillna(0.5)

    players["team_attack_raw"] = players[
        "team_attack_raw"
    ].fillna(players["team_attack_raw"].median())

    players["team_defence_raw"] = players[
        "team_defence_raw"
    ].fillna(players["team_defence_raw"].median())

    return players

# ============================================================
# FIXTURE DIFFICULTY
# ============================================================

def calculate_average_fdr(
    fixtures_df,
    teams_df,
):
    fixtures = fixtures_df.copy()

    output_columns = [
        "team",
        "average_fdr",
        "weighted_fdr",
        "next_fixture_fdr",
        "next_fixture_score",
        "fixture_score",
        "fixture_count",
        "next_fixtures",
    ]

    if fixtures.empty:
        return pd.DataFrame(columns=output_columns)

    remaining = fixtures[
        fixtures["finished"] == False
    ].copy()

    remaining["kickoff_time"] = pd.to_datetime(
        remaining["kickoff_time"],
        errors="coerce",
        utc=True,
    )

    remaining = remaining.sort_values(
        by=["kickoff_time", "event"],
        na_position="last",
    )

    team_map = (
        teams_df
        .set_index("id")["short_name"]
        .to_dict()
    )

    rows = []

    team_ids = sorted(
        set(remaining["team_h"].dropna().astype(int))
        | set(remaining["team_a"].dropna().astype(int))
    )

    for team_id in team_ids:
        team_fixtures = remaining[
            (remaining["team_h"] == team_id)
            | (remaining["team_a"] == team_id)
        ].head(UPCOMING_FIXTURE_COUNT)

        difficulties = []
        fixture_labels = []

        for _, fixture in team_fixtures.iterrows():
            if fixture["team_h"] == team_id:
                difficulty = float(fixture["team_h_difficulty"])
                opponent_id = int(fixture["team_a"])
                venue = "H"
            else:
                difficulty = float(fixture["team_a_difficulty"])
                opponent_id = int(fixture["team_h"])
                venue = "A"

            difficulties.append(difficulty)

            opponent = team_map.get(
                opponent_id,
                str(opponent_id),
            )

            fixture_labels.append(
                f"{opponent} ({venue}) [{int(difficulty)}]"
            )

        if not difficulties:
            continue

        raw_weights = FIXTURE_RECENCY_WEIGHTS[:len(difficulties)]

        # If the user later asks for more fixtures than weights supplied,
        # extend the tail with the final configured weight.
        if len(raw_weights) < len(difficulties):
            tail_weight = (
                FIXTURE_RECENCY_WEIGHTS[-1]
                if FIXTURE_RECENCY_WEIGHTS
                else 1.0
            )
            raw_weights = raw_weights + [
                tail_weight
            ] * (len(difficulties) - len(raw_weights))

        weights = np.array(raw_weights, dtype=float)
        weights = weights / weights.sum()

        weighted_fdr = float(
            np.average(
                np.array(difficulties, dtype=float),
                weights=weights,
            )
        )

        rows.append(
            {
                "team": team_id,
                "average_fdr": float(np.mean(difficulties)),
                "weighted_fdr": weighted_fdr,
                "next_fixture_fdr": float(difficulties[0]),
                "fixture_count": len(difficulties),
                "next_fixtures": " | ".join(fixture_labels),
            }
        )

    fdr = pd.DataFrame(rows)

    if fdr.empty:
        return pd.DataFrame(columns=output_columns)

    # Wildcard fixture score uses the configured multi-game horizon.
    # Lower difficulty is better.
    fdr["fixture_score"] = (
        1
        - minmax_scale_series(
            fdr["weighted_fdr"]
        )
    )

    # Captaincy uses ONLY the very next fixture.
    fdr["next_fixture_score"] = (
        1
        - minmax_scale_series(
            fdr["next_fixture_fdr"]
        )
    )

    return fdr


def add_fixture_strength(
    players,
    fixtures_df,
    teams_df,
):
    fdr = calculate_average_fdr(
        fixtures_df,
        teams_df,
    )

    players = players.merge(
        fdr,
        how="left",
        on="team",
    )

    players["fixture_score"] = players[
        "fixture_score"
    ].fillna(0.5)

    if players[
        "average_fdr"
    ].notna().any():
        median_fdr = players[
            "average_fdr"
        ].median()
    else:
        median_fdr = 3.0

    players["average_fdr"] = players[
        "average_fdr"
    ].fillna(median_fdr)

    if "weighted_fdr" not in players.columns:
        players["weighted_fdr"] = players["average_fdr"]

    players["weighted_fdr"] = players[
        "weighted_fdr"
    ].fillna(median_fdr)

    players["next_fixture_fdr"] = players[
        "next_fixture_fdr"
    ].fillna(median_fdr)

    players["next_fixture_score"] = players[
        "next_fixture_score"
    ].fillna(0.5)

    players["next_fixtures"] = players[
        "next_fixtures"
    ].fillna("")

    return players, fdr


# ============================================================
# RETENTION + CAPTAIN SCORING
# ============================================================

def add_retention_and_captain_scores(
    players,
):
    players = players.copy()

    component_sources = {
        "total_points":
            "total_points_num",
        "points_per_game":
            "points_per_game_num",
        "minutes_security":
            "minutes_security",
        "form":
            "form_num",
        "ict_index":
            "ict_index_num",
        "expected_goal_involvements":
            "expected_goal_involvements_num",
        "bonus":
            "bonus_num",
        "xgi_per_90":
            "xgi_per_90",
    }

    for (
        component_name,
        source_column,
    ) in component_sources.items():

        players[
            f"{component_name}_scaled"
        ] = minmax_scale_series(
            players[source_column]
        )

    players["retention_score"] = (
        weighted_score(
            players,
            RETENTION_WEIGHTS,
        )
        * 100
    )

    players["captain_score"] = (
        weighted_score(
            players,
            CAPTAIN_WEIGHTS,
        )
        * 100
    )

    return players


# ============================================================
# GAMEWEEK LEARNING HELPERS
# ============================================================

def normalize_target(series):
    return minmax_scale_series(
        pd.to_numeric(series, errors="coerce").fillna(0.0)
    )


def fit_nonnegative_simplex_weights(
    dataframe,
    feature_columns,
    target_column,
):
    """
    Fit one Gameweek as closely as possible:
      - weights >= 0
      - weights <= 1
      - weights sum to 1
      - minimize MSE against actual Gameweek points
    """
    working = dataframe[
        feature_columns + [target_column]
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if len(working) < MIN_PLAYERS_FOR_WEIGHT_FIT:
        return None, np.nan

    X = working[feature_columns].to_numpy(dtype=float)
    y = normalize_target(
        working[target_column]
    ).to_numpy(dtype=float)

    n_features = len(feature_columns)

    if n_features == 0:
        return None, np.nan

    initial = np.repeat(
        1.0 / n_features,
        n_features,
    )

    constraints = {
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1.0,
    }

    bounds = [
        (0.0, 1.0)
        for _ in range(n_features)
    ]

    def objective(weights):
        prediction = X @ weights
        return float(
            np.mean(
                (prediction - y) ** 2
            )
        )

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 5000,
            "ftol": 1e-12,
        },
    )

    if not result.success:
        return None, np.nan

    weights = np.clip(
        result.x,
        0.0,
        1.0,
    )

    if weights.sum() > 0:
        weights = (
            weights / weights.sum()
        )

    return (
        dict(
            zip(
                feature_columns,
                weights,
            )
        ),
        objective(weights),
    )


def calculate_rank_correlation(
    dataframe,
    predicted_column,
    actual_column,
):
    working = dataframe[
        [
            predicted_column,
            actual_column,
        ]
    ].dropna()

    if len(working) < 2:
        return np.nan

    predicted_rank = working[
        predicted_column
    ].rank(
        method="average",
        ascending=False,
    )

    actual_rank = working[
        actual_column
    ].rank(
        method="average",
        ascending=False,
    )

    return float(
        predicted_rank.corr(
            actual_rank
        )
    )


def fit_gameweek_weights(
    snapshot_players,
    gameweek,
    actual_points_column="actual_gw_points",
):
    """
    Fit a completed Gameweek separately by position,
    plus a separate captaincy model.

    snapshot_players must represent the features available
    BEFORE that Gameweek, with actual_gw_points added afterward.
    """
    if actual_points_column not in snapshot_players.columns:
        raise ValueError(
            f"Missing required column: {actual_points_column}"
        )

    rows = []

    for position, features in POSITION_LEARNING_FEATURES.items():
        subset = snapshot_players[
            snapshot_players["position"] == position
        ].copy()

        available = [
            feature
            for feature in features
            if feature in subset.columns
        ]

        weights, mse = fit_nonnegative_simplex_weights(
            subset,
            available,
            actual_points_column,
        )

        if weights is None:
            continue

        subset["_fit_score"] = 0.0

        for feature, weight in weights.items():
            subset["_fit_score"] += (
                subset[feature].fillna(0.0)
                * weight
            )

        rank_corr = calculate_rank_correlation(
            subset,
            "_fit_score",
            actual_points_column,
        )

        for feature, weight in weights.items():
            rows.append(
                {
                    "Gameweek": int(gameweek),
                    "Model": "POSITION",
                    "Position": position,
                    "Feature": feature,
                    "Weight": float(weight),
                    "Fit MSE": float(mse),
                    "Rank Correlation": rank_corr,
                    "Observations": int(len(subset)),
                }
            )

    captain_pool = snapshot_players[
        snapshot_players["position"].isin(
            ["midfielder", "forward"]
        )
    ].copy()

    captain_features = [
        feature
        for feature in CAPTAIN_LEARNING_FEATURES
        if feature in captain_pool.columns
    ]

    weights, mse = fit_nonnegative_simplex_weights(
        captain_pool,
        captain_features,
        actual_points_column,
    )

    if weights is not None:
        captain_pool["_fit_score"] = 0.0

        for feature, weight in weights.items():
            captain_pool["_fit_score"] += (
                captain_pool[feature].fillna(0.0)
                * weight
            )

        rank_corr = calculate_rank_correlation(
            captain_pool,
            "_fit_score",
            actual_points_column,
        )

        for feature, weight in weights.items():
            rows.append(
                {
                    "Gameweek": int(gameweek),
                    "Model": "CAPTAIN",
                    "Position": "captain",
                    "Feature": feature,
                    "Weight": float(weight),
                    "Fit MSE": float(mse),
                    "Rank Correlation": rank_corr,
                    "Observations": int(len(captain_pool)),
                }
            )

    return pd.DataFrame(rows)



def get_completed_gameweeks(fixtures_df):
    """
    Return completed FPL Gameweek numbers from the fixtures endpoint.
    """
    if fixtures_df.empty or "event" not in fixtures_df.columns:
        return []

    completed = (
        fixtures_df[
            fixtures_df["finished"] == True
        ]["event"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    return sorted(completed)


def get_next_unfinished_gameweek(fixtures_df):
    """
    Return the earliest Gameweek that has at least one unfinished fixture.
    """
    if fixtures_df.empty or "event" not in fixtures_df.columns:
        return None

    unfinished = fixtures_df[
        (fixtures_df["finished"] != True)
        & fixtures_df["event"].notna()
    ].copy()

    if unfinished.empty:
        return None

    return int(
        unfinished["event"]
        .astype(int)
        .min()
    )



def get_bootstrap_events():
    """
    Read FPL event metadata, including deadlines.
    """
    payload = get_json(FPL_BOOTSTRAP_URL)
    return pd.DataFrame(payload.get("events", []))


def get_gameweek_deadline_utc(events_df, gameweek):
    """
    Return the official FPL deadline as a timezone-aware UTC Timestamp.
    """
    if events_df.empty:
        return None

    row = events_df[
        events_df["id"].astype(int) == int(gameweek)
    ]

    if row.empty:
        return None

    deadline = row.iloc[0].get("deadline_time")

    if pd.isna(deadline) or deadline is None:
        return None

    return pd.to_datetime(
        deadline,
        utc=True,
    )


def utc_now():
    return pd.Timestamp.now(tz="UTC")


def safe_to_save_pre_gameweek_snapshot(
    events_df,
    gameweek,
):
    """
    A snapshot is valid only if it is written strictly BEFORE the official
    FPL deadline for that Gameweek.
    """
    deadline = get_gameweek_deadline_utc(
        events_df,
        gameweek,
    )

    if deadline is None:
        return False

    return utc_now() < deadline


def load_automation_state():
    path = Path(AUTOMATION_STATE_FILE)

    if not path.exists():
        return pd.DataFrame(
            columns=[
                "timestamp_utc",
                "action",
                "gameweek",
                "status",
                "details",
            ]
        )

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(
            columns=[
                "timestamp_utc",
                "action",
                "gameweek",
                "status",
                "details",
            ]
        )


def log_automation_action(
    action,
    gameweek,
    status,
    details="",
):
    state = load_automation_state()

    row = pd.DataFrame(
        [
            {
                "timestamp_utc": utc_now().isoformat(),
                "action": action,
                "gameweek": (
                    int(gameweek)
                    if gameweek is not None
                    else None
                ),
                "status": status,
                "details": details,
            }
        ]
    )

    state = pd.concat(
        [state, row],
        ignore_index=True,
    )

    state.to_csv(
        AUTOMATION_STATE_FILE,
        index=False,
    )


def snapshot_path(gameweek):
    directory = Path(SNAPSHOT_DIRECTORY)
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory / f"pre_gw{int(gameweek)}_snapshot.csv"


def build_snapshot_columns(players):
    """
    Keep identifiers plus every feature needed by the learning models.
    """
    base_columns = [
        "id",
        "player_name",
        "team",
        "team_name",
        "position",
        "now_cost",
    ]

    learning_columns = set()

    for features in POSITION_LEARNING_FEATURES.values():
        learning_columns.update(features)

    learning_columns.update(
        CAPTAIN_LEARNING_FEATURES
    )

    ordered = []

    for column in base_columns + sorted(learning_columns):
        if (
            column in players.columns
            and column not in ordered
        ):
            ordered.append(column)

    return ordered


def save_pre_gameweek_snapshot(
    players,
    gameweek,
    events_df=None,
    refresh_before_deadline=True,
):
    """
    Persist the PRE-GW feature state.

    Autonomous behaviour:
    - before the deadline: repeated scheduled runs REFRESH the same GW snapshot,
      so the final saved file is the freshest state observed before deadline.
    - at/after the deadline: the file is frozen forever and cannot be modified.

    This removes the need for the user to remember to save the final snapshot.
    """
    path = snapshot_path(gameweek)

    if events_df is not None:
        deadline = get_gameweek_deadline_utc(
            events_df,
            gameweek,
        )

        if deadline is None:
            print(
                f"GW{gameweek}: deadline unavailable; "
                "snapshot not written."
            )
            log_automation_action(
                "snapshot",
                gameweek,
                "skipped",
                "Deadline unavailable",
            )
            return path if path.exists() else None

        if utc_now() >= deadline:
            print(
                f"GW{gameweek}: deadline has passed; "
                "snapshot is frozen."
            )
            log_automation_action(
                "snapshot",
                gameweek,
                "frozen",
                f"Deadline {deadline.isoformat()}",
            )
            return path if path.exists() else None

    if path.exists() and not refresh_before_deadline:
        print(
            f"Pre-GW{gameweek} snapshot already exists: {path}"
        )
        return path

    columns = build_snapshot_columns(
        players
    )

    snapshot = players[
        columns
    ].copy()

    snapshot["snapshot_gameweek"] = int(gameweek)
    snapshot["snapshot_saved_utc"] = (
        utc_now().isoformat()
    )

    if events_df is not None:
        deadline = get_gameweek_deadline_utc(
            events_df,
            gameweek,
        )
        snapshot["deadline_utc"] = (
            deadline.isoformat()
            if deadline is not None
            else None
        )

    # Atomic write: write temporary file first, then replace.
    temp_path = path.with_suffix(".tmp.csv")

    snapshot.to_csv(
        temp_path,
        index=False,
    )

    temp_path.replace(path)

    print(
        f"Saved/refreshed PRE-GW{gameweek} snapshot: "
        f"{path} ({len(snapshot)} players)"
    )

    log_automation_action(
        "snapshot",
        gameweek,
        "saved",
        f"{len(snapshot)} players",
    )

    return path


def fetch_player_gameweek_points(
    player_ids,
    gameweek,
):
    """
    Fetch actual FPL points for a completed Gameweek from each player's
    element-summary history.
    """
    rows = []

    total = len(player_ids)

    for index, player_id in enumerate(
        player_ids,
        start=1,
    ):
        try:
            url = (
                FPL_ELEMENT_SUMMARY_URL
                .format(
                    player_id=int(player_id)
                )
            )

            payload = get_json(url)

            history = payload.get(
                "history",
                [],
            )

            gw_row = next(
                (
                    row
                    for row in history
                    if int(
                        row.get(
                            "round",
                            -1,
                        )
                    ) == int(gameweek)
                ),
                None,
            )

            actual_points = (
                float(
                    gw_row.get(
                        "total_points",
                        0.0,
                    )
                )
                if gw_row is not None
                else 0.0
            )

            rows.append(
                {
                    "id": int(player_id),
                    "actual_gw_points": actual_points,
                }
            )

        except Exception:
            rows.append(
                {
                    "id": int(player_id),
                    "actual_gw_points": 0.0,
                }
            )

        if (
            index % 100 == 0
            or index == total
        ):
            print(
                f"  GW{gameweek}: fetched actual points "
                f"for {index}/{total} players"
            )

    return pd.DataFrame(rows)


def learn_completed_gameweek_from_snapshot(
    gameweek,
):
    """
    Learn one completed GW ONLY from its saved pre-match snapshot.
    """
    path = snapshot_path(
        gameweek
    )

    if not path.exists():
        print(
            f"GW{gameweek}: no pre-match snapshot found; "
            "skipping learning."
        )
        return pd.DataFrame()

    snapshot = pd.read_csv(
        path
    )

    if "id" not in snapshot.columns:
        raise ValueError(
            f"Snapshot {path} does not contain player id."
        )

    player_ids = (
        snapshot["id"]
        .dropna()
        .astype(int)
        .tolist()
    )

    actual_points = (
        fetch_player_gameweek_points(
            player_ids,
            gameweek,
        )
    )

    learning_frame = snapshot.merge(
        actual_points,
        how="left",
        on="id",
    )

    learning_frame[
        "actual_gw_points"
    ] = (
        learning_frame[
            "actual_gw_points"
        ]
        .fillna(0.0)
    )

    gw_weights = fit_gameweek_weights(
        learning_frame,
        gameweek=gameweek,
        actual_points_column="actual_gw_points",
    )

    if gw_weights.empty:
        print(
            f"GW{gameweek}: no weights could be fitted."
        )
        return gw_weights

    save_gameweek_weights(
        gw_weights
    )

    print(
        f"GW{gameweek}: learned from PRE-GW{gameweek} "
        f"snapshot and saved {len(gw_weights)} weights."
    )

    log_automation_action(
        "learn",
        gameweek,
        "completed",
        f"{len(gw_weights)} learned weights",
    )

    return gw_weights


def auto_learn_from_saved_snapshots(
    fixtures_df,
):
    """
    Learn only completed Gameweeks for which a pre-match snapshot exists.

    Old completed GWs are NOT reconstructed from current/post-match data.
    """
    history = load_weight_history()

    completed_gameweeks = get_completed_gameweeks(
        fixtures_df
    )

    already_learned = set()

    if not history.empty:
        already_learned = set(
            history["Gameweek"]
            .dropna()
            .astype(int)
            .tolist()
        )

    candidates = [
        gw
        for gw in completed_gameweeks
        if gw not in already_learned
        and snapshot_path(gw).exists()
    ]

    for gameweek in candidates:
        learn_completed_gameweek_from_snapshot(
            gameweek
        )

    return load_weight_history()


def load_weight_history():
    path = Path(WEIGHT_HISTORY_FILE)

    if not path.exists():
        return pd.DataFrame(
            columns=[
                "Gameweek",
                "Model",
                "Position",
                "Feature",
                "Weight",
                "Fit MSE",
                "Rank Correlation",
                "Observations",
            ]
        )

    return pd.read_csv(path)


def save_gameweek_weights(gameweek_weights):
    history = load_weight_history()

    if gameweek_weights.empty:
        return history

    gameweeks = (
        gameweek_weights["Gameweek"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    if not history.empty and gameweeks:
        history = history[
            ~history["Gameweek"]
            .astype("Int64")
            .isin(gameweeks)
        ].copy()

    combined = pd.concat(
        [
            history,
            gameweek_weights,
        ],
        ignore_index=True,
    )

    combined = combined.sort_values(
        by=[
            "Gameweek",
            "Model",
            "Position",
            "Feature",
        ]
    ).reset_index(drop=True)

    combined.to_csv(
        WEIGHT_HISTORY_FILE,
        index=False,
    )

    return combined


def summarize_weight_history(history):
    """
    Calculate mean, median and next-GW estimate.
    Next-GW estimate = MEDIAN.
    """
    columns = [
        "Model",
        "Position",
        "Feature",
        "Gameweeks",
        "Mean Weight",
        "Median Weight",
        "Std Weight",
        "Min Weight",
        "Max Weight",
        "Next GW Estimate",
    ]

    if history.empty:
        empty = pd.DataFrame(columns=columns)
        return empty, empty.copy()

    summary = (
        history.groupby(
            [
                "Model",
                "Position",
                "Feature",
            ],
            as_index=False,
        )
        .agg(
            Gameweeks=("Gameweek", "nunique"),
            Mean_Weight=("Weight", "mean"),
            Median_Weight=("Weight", "median"),
            Std_Weight=("Weight", "std"),
            Min_Weight=("Weight", "min"),
            Max_Weight=("Weight", "max"),
        )
    )

    summary = summary.rename(
        columns={
            "Mean_Weight": "Mean Weight",
            "Median_Weight": "Median Weight",
            "Std_Weight": "Std Weight",
            "Min_Weight": "Min Weight",
            "Max_Weight": "Max Weight",
        }
    )

    summary["Std Weight"] = (
        summary["Std Weight"]
        .fillna(0.0)
    )

    summary["Next GW Estimate"] = (
        summary["Median Weight"]
    )

    summary = summary[
        columns
    ].sort_values(
        by=[
            "Model",
            "Position",
            "Next GW Estimate",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    ).reset_index(drop=True)

    next_weights = summary[
        [
            "Model",
            "Position",
            "Feature",
            "Gameweeks",
            "Mean Weight",
            "Median Weight",
            "Next GW Estimate",
        ]
    ].copy()

    return summary, next_weights


def median_weight_map(
    history,
    model,
    position,
):
    subset = history[
        (history["Model"] == model)
        & (
            history["Position"]
            == position
        )
    ].copy()

    if subset.empty:
        return {}

    medians = (
        subset.groupby(
            "Feature"
        )["Weight"]
        .median()
    )

    if medians.sum() > 0:
        medians = (
            medians / medians.sum()
        )

    return medians.to_dict()


def apply_learned_median_scores(
    players,
    history,
):
    output = players.copy()

    output["learned_median_score"] = np.nan
    output["learned_median_captain_score"] = np.nan

    for position in POSITION_LEARNING_FEATURES:
        weights = median_weight_map(
            history,
            "POSITION",
            position,
        )

        if not weights:
            continue

        mask = (
            output["position"]
            == position
        )

        score = pd.Series(
            0.0,
            index=output.index,
        )

        for feature, weight in weights.items():
            if feature in output.columns:
                score += (
                    output[feature]
                    .fillna(0.0)
                    * weight
                )

        output.loc[
            mask,
            "learned_median_score",
        ] = (
            score.loc[mask]
            * 100.0
        )

    captain_weights = median_weight_map(
        history,
        "CAPTAIN",
        "captain",
    )

    if captain_weights:
        captain_score = pd.Series(
            0.0,
            index=output.index,
        )

        for feature, weight in captain_weights.items():
            if feature in output.columns:
                captain_score += (
                    output[feature]
                    .fillna(0.0)
                    * weight
                )

        output[
            "learned_median_captain_score"
        ] = (
            captain_score * 100.0
        )

    return output


# ============================================================
# PCA + K-MEANS SEGMENTATION
# ============================================================

def add_player_segments(players):
    """
    PCA/K-Means is descriptive segmentation only.
    It does NOT remove players from Wildcard optimization.
    """
    players = players.copy()

    segmentation_features = [
        "total_points_scaled",
        "points_per_game_scaled",
        "minutes_security_scaled",
        "form_scaled",
        "ict_index_scaled",
        "expected_goal_involvements_scaled",
        "xgi_per_90_scaled",
        "bonus_scaled",
    ]

    segment_data = players[
        segmentation_features
    ].fillna(0.0)

    if len(segment_data) < N_CLUSTERS:
        players["cluster"] = 0
        players["cluster_quality"] = players[
            "retention_score"
        ].mean()
        players["cluster_label"] = "Core"
        return players

    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(
        segment_data
    )

    n_components = min(
        4,
        scaled.shape[1],
        scaled.shape[0],
    )

    pca = PCA(
        n_components=n_components,
        random_state=RANDOM_STATE,
    )

    reduced = pca.fit_transform(
        scaled
    )

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        n_init=20,
        random_state=RANDOM_STATE,
    )

    players["cluster"] = (
        kmeans.fit_predict(
            reduced
        )
    )

    cluster_quality = (
        players.groupby(
            "cluster"
        )["retention_score"]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    ordered_clusters = (
        cluster_quality
        .index
        .tolist()
    )

    labels = {}

    if len(ordered_clusters) >= 1:
        labels[
            ordered_clusters[0]
        ] = "Core"

    if len(ordered_clusters) >= 2:
        labels[
            ordered_clusters[1]
        ] = "Watch"

    if len(ordered_clusters) >= 3:
        for cluster_id in ordered_clusters[2:]:
            labels[
                cluster_id
            ] = "Avoid"

    players["cluster_quality"] = (
        players["cluster"].map(
            cluster_quality.to_dict()
        )
    )

    players["cluster_label"] = (
        players["cluster"].map(
            labels
        )
    )

    return players


# ============================================================
# FINAL SCORE
# ============================================================

def add_final_score(players):
    """
    V6 final score:
      GK/DEF -> individual retention + current-club defence + next fixtures
      MID/FWD -> individual retention + current-club attack + next fixtures

    Captain Score:
      individual attacking ceiling + current-club attack + next fixtures
    """
    players = players.copy()

    retention_component = (
        players["retention_score"] / 100.0
    )

    # Preserve the pre-context captain score for diagnostics.
    players["individual_captain_ceiling"] = (
        players["captain_score"] / 100.0
    )

    players["team_context_score"] = np.where(
        players["position"].isin(
            ["goalkeeper", "defender"]
        ),
        players["team_defence_score"],
        players["team_attack_score"],
    )

    players["final_score"] = 0.0

    for position, weights in POSITION_SCORE_WEIGHTS.items():
        mask = players["position"] == position

        players.loc[mask, "final_score"] = (
            (
                weights["retention"]
                * retention_component.loc[mask]
            )
            + (
                weights["team_context"]
                * players.loc[mask, "team_context_score"]
            )
            + (
                weights["fixtures"]
                * players.loc[mask, "fixture_score"]
            )
        ) * 100.0

    # Captaincy is intentionally a SINGLE-GAME decision.
    # fixture_score is NOT used here because that represents the 3-game
    # Wildcard horizon. Only next_fixture_score is used.
    players["captain_score"] = (
        (
            CAPTAIN_SINGLE_FIXTURE_WEIGHTS["next_fixture"]
            * players["next_fixture_score"]
        )
        + (
            CAPTAIN_SINGLE_FIXTURE_WEIGHTS["individual_ceiling"]
            * players["individual_captain_ceiling"]
        )
        + (
            CAPTAIN_SINGLE_FIXTURE_WEIGHTS["team_attack"]
            * players["team_attack_score"]
        )
        + (
            CAPTAIN_SINGLE_FIXTURE_WEIGHTS["minutes_security"]
            * players["minutes_security"]
        )
    ) * 100.0

    return players


# ============================================================
# POSITION RANKINGS
# ============================================================

def build_top_players(
    players,
    position,
    count,
):
    ranked = players[
        players["position"] == position
    ].copy()

    ranked = ranked.sort_values(
        by=[
            "final_score",
            "retention_score",
            "captain_score",
        ],
        ascending=False,
    )

    columns = [
        "id",
        "web_name",
        "team_name",
        "position",
        "price",
        "final_score",
        "retention_score",
        "captain_score",
        "learned_median_score",
        "learned_median_captain_score",
        "cluster_label",
        "total_points_num",
        "points_per_game_num",
        "form_num",
        "minutes_num",
        "value",
        "average_fdr",
        "weighted_fdr",
        "next_fixture_fdr",
        "next_fixture_score",
        "fixture_score",
        "team_context_score",
        "team_attack_score",
        "team_defence_score",
        "expected_goal_involvements_num",
        "xgi_per_90",
        "next_fixtures",
    ]

    output = ranked.head(
        count
    )[columns].copy()

    return format_player_output(output)


def format_player_output(df):
    output = df.copy()

    output = output.rename(
        columns={
            "id": "ID",
            "web_name": "Player",
            "team_name": "Team",
            "position": "Position",
            "price": "Price",
            "final_score": "Final Score",
            "retention_score": "Retention",
            "captain_score": "Captain Score",
            "learned_median_score": "Median-Learned Score",
            "learned_median_captain_score": "Median-Learned Captain",
            "cluster_label": "Segment",
            "total_points_num": "Points",
            "points_per_game_num": "PPG",
            "form_num": "Form",
            "minutes_num": "Minutes",
            "value": "Pts/£m",
            "average_fdr": "Avg FDR",
            "weighted_fdr": "Weighted FDR",
            "next_fixture_fdr": "Next FDR",
            "next_fixture_score": "Next Fixture Score",
            "fixture_score": "3-Fixture Score",
            "team_context_score": "Team Context",
            "team_attack_score": "Team Attack",
            "team_defence_score": "Team Defence",
            "expected_goal_involvements_num": "xGI",
            "xgi_per_90": "xGI/90",
            "next_fixtures": "Next Fixtures",
        }
    )

    numeric_columns = [
        "Price",
        "Final Score",
        "Retention",
        "Captain Score",
        "Median-Learned Score",
        "Median-Learned Captain",
        "Points",
        "PPG",
        "Form",
        "Minutes",
        "Pts/£m",
        "Avg FDR",
        "Weighted FDR",
        "Next FDR",
        "Next Fixture Score",
        "3-Fixture Score",
        "Team Context",
        "Team Attack",
        "Team Defence",
        "xGI",
        "xGI/90",
    ]

    for column in numeric_columns:
        if column in output.columns:
            output[column] = pd.to_numeric(
                output[column],
                errors="coerce",
            ).round(2)

    return output.reset_index(
        drop=True
    )


# ============================================================
# WILDCARD OPTIMIZER
# ============================================================

def optimize_wildcard(players):
    """
    Optimize the Wildcard in two stages.

    Stage 1:
        Maximize Starting XI quality + Captain ceiling.

    Stage 2:
        Hold that optimal football-quality result constant and minimize
        the total cost of the four bench players.

    This prevents the optimizer from wasting budget on substitutes that are
    unlikely to score points while keeping the strongest feasible XI.

    IMPORTANT:
    All eligible players are considered regardless of Core/Watch/Avoid.
    """

    pool = players.copy().reset_index(drop=True)

    pool = pool[
        pool["final_score"].notna()
        & pool["captain_score"].notna()
        & pool["now_cost_num"].notna()
        & pool["position"].notna()
        & pool["team"].notna()
    ].copy().reset_index(drop=True)

    if pool.empty:
        raise RuntimeError(
            "No eligible players are available for Wildcard optimization."
        )

    n = len(pool)
    total_vars = 3 * n

    scores = pool["final_score"].to_numpy(dtype=float)
    captain_scores = pool["captain_score"].to_numpy(dtype=float)
    costs = pool["now_cost_num"].to_numpy(dtype=float)

    # Variables:
    # y_i = player in 15-man squad
    # x_i = player in starting XI
    # c_i = player is captain
    #
    # Vector = [y, x, c]

    rows = []
    lower_bounds = []
    upper_bounds = []

    def add_constraint(coefficients, lower, upper):
        rows.append(coefficients)
        lower_bounds.append(lower)
        upper_bounds.append(upper)

    # --------------------------------------------------------
    # Squad size: exactly 15
    # --------------------------------------------------------
    coeff = np.zeros(total_vars)
    coeff[:n] = 1
    add_constraint(coeff, 15, 15)

    # --------------------------------------------------------
    # Full squad positional requirements
    # --------------------------------------------------------
    for position, required_count in SQUAD_POSITION_COUNTS.items():
        coeff = np.zeros(total_vars)
        mask = (pool["position"] == position).to_numpy()
        coeff[:n][mask] = 1

        add_constraint(
            coeff,
            required_count,
            required_count,
        )

    # --------------------------------------------------------
    # £100m budget
    # --------------------------------------------------------
    coeff = np.zeros(total_vars)
    coeff[:n] = costs

    add_constraint(
        coeff,
        -np.inf,
        WILDCARD_BUDGET,
    )

    # --------------------------------------------------------
    # Maximum three players per club
    # --------------------------------------------------------
    for team_id in sorted(pool["team"].dropna().unique()):
        coeff = np.zeros(total_vars)
        mask = (pool["team"] == team_id).to_numpy()
        coeff[:n][mask] = 1

        add_constraint(
            coeff,
            -np.inf,
            MAX_PLAYERS_PER_CLUB,
        )

    # --------------------------------------------------------
    # Exactly 11 starters
    # --------------------------------------------------------
    coeff = np.zeros(total_vars)
    coeff[n:2*n] = 1
    add_constraint(coeff, 11, 11)

    # --------------------------------------------------------
    # Exactly one captain
    # --------------------------------------------------------
    coeff = np.zeros(total_vars)
    coeff[2*n:] = 1
    add_constraint(coeff, 1, 1)

    # --------------------------------------------------------
    # Legal starting formation
    # --------------------------------------------------------
    formation_minimums = {
        "goalkeeper": (1, 1),
        "defender": (3, np.inf),
        "midfielder": (2, np.inf),
        "forward": (1, np.inf),
    }

    for position, (lower, upper) in formation_minimums.items():
        coeff = np.zeros(total_vars)
        mask = (pool["position"] == position).to_numpy()
        coeff[n:2*n][mask] = 1

        add_constraint(
            coeff,
            lower,
            upper,
        )

    # --------------------------------------------------------
    # Linking constraints
    # --------------------------------------------------------
    for i in range(n):
        # Starter must belong to squad.
        # x_i <= y_i
        coeff = np.zeros(total_vars)
        coeff[i] = -1
        coeff[n + i] = 1
        add_constraint(coeff, -np.inf, 0)

        # Captain must be a starter.
        # c_i <= x_i
        coeff = np.zeros(total_vars)
        coeff[n + i] = -1
        coeff[2*n + i] = 1
        add_constraint(coeff, -np.inf, 0)

    def build_constraint_matrix(extra_rows=None):
        local_rows = list(rows)
        local_lower = list(lower_bounds)
        local_upper = list(upper_bounds)

        if extra_rows:
            for row, lower, upper in extra_rows:
                local_rows.append(row)
                local_lower.append(lower)
                local_upper.append(upper)

        A = lil_matrix(
            (len(local_rows), total_vars),
            dtype=float,
        )

        for row_index, row in enumerate(local_rows):
            nonzero = np.nonzero(row)[0]
            A[row_index, nonzero] = row[nonzero]

        return LinearConstraint(
            A.tocsr(),
            np.array(local_lower, dtype=float),
            np.array(local_upper, dtype=float),
        )

    bounds = Bounds(
        lb=np.zeros(total_vars),
        ub=np.ones(total_vars),
    )

    integrality = np.ones(
        total_vars,
        dtype=int,
    )

    # ========================================================
    # STAGE 1: maximize starting XI + captain
    # ========================================================
    #
    # scipy.optimize.milp minimizes, so negate football quality.
    stage1_objective = np.zeros(
        total_vars,
        dtype=float,
    )

    stage1_objective[n:2*n] = -(
        STARTER_WEIGHT * scores
    )

    stage1_objective[2*n:] = -(
        captain_scores
    )

    stage1_result = milp(
        c=stage1_objective,
        integrality=integrality,
        bounds=bounds,
        constraints=build_constraint_matrix(),
        options={"disp": False},
    )

    if not stage1_result.success:
        raise RuntimeError(
            "Wildcard Stage 1 optimization failed: "
            f"{stage1_result.message}"
        )

    optimal_football_quality = -float(
        stage1_result.fun
    )

    # ========================================================
    # STAGE 2: keep optimal XI quality, minimize bench cost
    # ========================================================
    #
    # Bench indicator = y_i - x_i.
    #
    # Minimize:
    #     sum(cost_i * y_i) - sum(cost_i * x_i)
    #
    # Starter cost therefore cancels completely.
    # Only the four substitutes are priced by this objective.
    stage2_objective = np.zeros(
        total_vars,
        dtype=float,
    )

    stage2_objective[:n] = costs
    stage2_objective[n:2*n] = -costs

    # Football quality constraint:
    # sum(final_score_i * x_i)
    # + sum(captain_score_i * c_i)
    # >= Stage 1 optimum
    quality_row = np.zeros(
        total_vars,
        dtype=float,
    )

    quality_row[n:2*n] = (
        STARTER_WEIGHT * scores
    )

    quality_row[2*n:] = captain_scores

    quality_floor = (
        optimal_football_quality
        - BENCH_OPTIMALITY_TOLERANCE
    )

    stage2_constraints = build_constraint_matrix(
        extra_rows=[
            (
                quality_row,
                quality_floor,
                np.inf,
            )
        ]
    )

    stage2_result = milp(
        c=stage2_objective,
        integrality=integrality,
        bounds=bounds,
        constraints=stage2_constraints,
        options={"disp": False},
    )

    if not stage2_result.success:
        raise RuntimeError(
            "Wildcard Stage 2 optimization failed: "
            f"{stage2_result.message}"
        )

    solution = np.rint(
        stage2_result.x
    ).astype(int)

    squad_selected = solution[:n].astype(bool)
    starters_selected = solution[n:2*n].astype(bool)
    captain_selected = solution[2*n:].astype(bool)

    selected_indices = np.where(
        squad_selected
    )[0]

    squad = pool.loc[
        squad_selected
    ].copy()

    squad["Starting XI"] = (
        starters_selected[
            selected_indices
        ]
    )

    squad["Captain"] = (
        captain_selected[
            selected_indices
        ]
    )

    # --------------------------------------------------------
    # Captain
    # --------------------------------------------------------
    captain_id = None

    if squad["Captain"].any():
        captain_id = squad.loc[
            squad["Captain"],
            "id",
        ].iloc[0]

    # --------------------------------------------------------
    # Vice captain
    # --------------------------------------------------------
    vice_pool = squad[
        squad["Starting XI"]
    ].copy()

    if captain_id is not None:
        vice_pool = vice_pool[
            vice_pool["id"] != captain_id
        ]

    vice_pool = vice_pool.sort_values(
        by=[
            "captain_score",
            "final_score",
        ],
        ascending=False,
    )

    vice_id = None

    if not vice_pool.empty:
        vice_id = vice_pool.iloc[0]["id"]

    squad["Role"] = np.where(
        squad["Starting XI"],
        "START",
        "BENCH",
    )

    if captain_id is not None:
        squad.loc[
            squad["id"] == captain_id,
            "Role",
        ] = "CAPTAIN"

    if vice_id is not None:
        squad.loc[
            squad["id"] == vice_id,
            "Role",
        ] = "VICE"

    # --------------------------------------------------------
    # Bench order
    # --------------------------------------------------------
    bench = squad[
        ~squad["Starting XI"]
    ].copy()

    bench_gk = bench[
        bench["position"] == "goalkeeper"
    ].copy()

    bench_outfield = bench[
        bench["position"] != "goalkeeper"
    ].copy()

    # Since bench cost is deliberately minimized, order the outfield
    # substitutes by quality so the strongest cheap reserve is first.
    bench_outfield = bench_outfield.sort_values(
        by=[
            "final_score",
            "minutes_security",
        ],
        ascending=False,
    )

    for order, (_, row) in enumerate(
        bench_outfield.iterrows(),
        start=1,
    ):
        squad.loc[
            squad["id"] == row["id"],
            "Role",
        ] = f"BENCH {order}"

    for _, row in bench_gk.iterrows():
        squad.loc[
            squad["id"] == row["id"],
            "Role",
        ] = "BENCH GK"

    # --------------------------------------------------------
    # Output sorting
    # --------------------------------------------------------
    position_order = {
        "goalkeeper": 1,
        "defender": 2,
        "midfielder": 3,
        "forward": 4,
    }

    role_order = {
        "CAPTAIN": 1,
        "VICE": 2,
        "START": 3,
        "BENCH 1": 4,
        "BENCH 2": 5,
        "BENCH 3": 6,
        "BENCH GK": 7,
    }

    squad["position_order"] = (
        squad["position"].map(
            position_order
        )
    )

    squad["role_order"] = (
        squad["Role"].map(
            role_order
        ).fillna(99)
    )

    squad = squad.sort_values(
        by=[
            "Starting XI",
            "position_order",
            "role_order",
            "final_score",
        ],
        ascending=[
            False,
            True,
            True,
            False,
        ],
    )

    total_cost = squad[
        "price"
    ].sum()

    starter_score = squad.loc[
        squad["Starting XI"],
        "final_score",
    ].sum()

    bench_score = squad.loc[
        ~squad["Starting XI"],
        "final_score",
    ].sum()

    bench_cost = squad.loc[
        ~squad["Starting XI"],
        "price",
    ].sum()

    captain_bonus = squad.loc[
        squad["Captain"],
        "captain_score",
    ].sum()

    objective_score = (
        starter_score
        + captain_bonus
    )

    return (
        squad,
        total_cost,
        starter_score,
        bench_score,
        bench_cost,
        captain_bonus,
        objective_score,
    )


def build_wildcard_output(squad):
    columns = [
        "id",
        "web_name",
        "team_name",
        "position",
        "Role",
        "price",
        "final_score",
        "retention_score",
        "captain_score",
        "learned_median_score",
        "learned_median_captain_score",
        "cluster_label",
        "total_points_num",
        "points_per_game_num",
        "form_num",
        "minutes_num",
        "value",
        "average_fdr",
        "weighted_fdr",
        "next_fixture_fdr",
        "next_fixture_score",
        "fixture_score",
        "team_context_score",
        "team_attack_score",
        "team_defence_score",
        "expected_goal_involvements_num",
        "xgi_per_90",
        "next_fixtures",
    ]

    output = squad[
        columns
    ].copy()

    output = output.rename(
        columns={
            "id": "ID",
            "web_name": "Player",
            "team_name": "Team",
            "position": "Position",
            "price": "Price",
            "final_score": "Final Score",
            "retention_score": "Retention",
            "captain_score": "Captain Score",
            "learned_median_score": "Median-Learned Score",
            "learned_median_captain_score": "Median-Learned Captain",
            "cluster_label": "Segment",
            "total_points_num": "Points",
            "points_per_game_num": "PPG",
            "form_num": "Form",
            "minutes_num": "Minutes",
            "value": "Pts/£m",
            "average_fdr": "Avg FDR",
            "weighted_fdr": "Weighted FDR",
            "next_fixture_fdr": "Next FDR",
            "next_fixture_score": "Next Fixture Score",
            "fixture_score": "3-Fixture Score",
            "team_context_score": "Team Context",
            "team_attack_score": "Team Attack",
            "team_defence_score": "Team Defence",
            "expected_goal_involvements_num": "xGI",
            "xgi_per_90": "xGI/90",
            "next_fixtures": "Next Fixtures",
        }
    )

    numeric_columns = [
        "Price",
        "Final Score",
        "Retention",
        "Captain Score",
        "Median-Learned Score",
        "Median-Learned Captain",
        "Points",
        "PPG",
        "Form",
        "Minutes",
        "Pts/£m",
        "Avg FDR",
        "Weighted FDR",
        "Next FDR",
        "Next Fixture Score",
        "3-Fixture Score",
        "Team Context",
        "Team Attack",
        "Team Defence",
        "xGI",
        "xGI/90",
    ]

    for column in numeric_columns:
        output[column] = pd.to_numeric(
            output[column],
            errors="coerce",
        ).round(2)

    return output.reset_index(
        drop=True
    )


# ============================================================
# EXCEL OUTPUT
# ============================================================

def write_excel_output(
    players,
    fdr,
    top_goalkeepers,
    top_defenders,
    top_midfielders,
    top_forwards,
    wildcard_squad,
    total_cost,
    starter_score,
    bench_score,
    bench_cost,
    captain_bonus,
    objective_score,
    weight_history,
    weight_summary,
    next_gw_weights,
):
    wildcard_output = build_wildcard_output(
        wildcard_squad
    )

    starting_xi = wildcard_output[
        ~wildcard_output["Role"]
        .str.startswith("BENCH")
    ].copy()

    bench = wildcard_output[
        wildcard_output["Role"]
        .str.startswith("BENCH")
    ].copy()

    captain_rows = wildcard_output[
        wildcard_output["Role"]
        == "CAPTAIN"
    ]

    vice_rows = wildcard_output[
        wildcard_output["Role"]
        == "VICE"
    ]

    captain_name = (
        captain_rows.iloc[0]["Player"]
        if not captain_rows.empty
        else ""
    )

    vice_name = (
        vice_rows.iloc[0]["Player"]
        if not vice_rows.empty
        else ""
    )

    summary = pd.DataFrame(
        {
            "Metric": [
                "Budget Used",
                "Budget Limit",
                "Starting XI Score",
                "Captain Bonus",
                "Bench Score (Informational)",
                "Bench Cost",
                "Optimized XI + Captain Objective",
                "Captain",
                "Vice Captain",
                "Player Retention Weight",
                "Current Club Context Weight",
                "Fixture Weight",
                "Fixtures Used for Wildcard",
                "Captain Fixture Horizon",
                "Captain Next-Fixture Weight",
                "Captain Ceiling Weight",
                "Captain Team Attack Weight",
                "Captain Minutes Weight",
                "Starter Weight",
                "Bench Strategy",
                "Completed Snapshot GWs Learned",
                "Next Weight Estimator",
            ],
            "Value": [
                f"£{total_cost:.1f}m",
                f"£{WILDCARD_BUDGET / 10:.1f}m",
                round(
                    starter_score,
                    2,
                ),
                round(
                    captain_bonus,
                    2,
                ),
                round(
                    bench_score,
                    2,
                ),
                f"£{bench_cost:.1f}m",
                round(
                    objective_score,
                    2,
                ),
                captain_name,
                vice_name,
                f"{POSITION_SCORE_WEIGHTS['forward']['retention']:.0%}",
                f"{POSITION_SCORE_WEIGHTS['forward']['team_context']:.0%}",
                f"{POSITION_SCORE_WEIGHTS['forward']['fixtures']:.0%}",
                UPCOMING_FIXTURE_COUNT,
                "1 fixture only",
                f"{CAPTAIN_SINGLE_FIXTURE_WEIGHTS['next_fixture']:.0%}",
                f"{CAPTAIN_SINGLE_FIXTURE_WEIGHTS['individual_ceiling']:.0%}",
                f"{CAPTAIN_SINGLE_FIXTURE_WEIGHTS['team_attack']:.0%}",
                f"{CAPTAIN_SINGLE_FIXTURE_WEIGHTS['minutes_security']:.0%}",
                f"{STARTER_WEIGHT:.0%}",
                "Cheapest feasible 4-player bench",
                (
                    int(weight_history["Gameweek"].nunique())
                    if not weight_history.empty
                    else 0
                ),
                NEXT_WEIGHT_ESTIMATOR,
            ],
        }
    )

    all_columns = [
        "id",
        "web_name",
        "team_name",
        "position",
        "price",
        "final_score",
        "retention_score",
        "captain_score",
        "learned_median_score",
        "learned_median_captain_score",
        "cluster_label",
        "total_points_num",
        "points_per_game_num",
        "form_num",
        "minutes_num",
        "value",
        "average_fdr",
        "weighted_fdr",
        "next_fixture_fdr",
        "next_fixture_score",
        "fixture_score",
        "team_context_score",
        "team_attack_score",
        "team_defence_score",
        "expected_goal_involvements_num",
        "xgi_per_90",
        "next_fixtures",
    ]

    all_players = format_player_output(
        players[
            all_columns
        ]
        .sort_values(
            by="final_score",
            ascending=False,
        )
    )

    fixture_output = fdr.merge(
        players[
            [
                "team",
                "team_name",
            ]
        ]
        .drop_duplicates(),
        how="left",
        on="team",
    )

    fixture_output = fixture_output[
        [
            "team_name",
            "average_fdr",
            "weighted_fdr",
            "next_fixture_fdr",
            "next_fixture_score",
            "fixture_score",
            "fixture_count",
            "next_fixtures",
        ]
    ].rename(
        columns={
            "team_name": "Team",
            "average_fdr": "Avg FDR",
            "weighted_fdr": "Weighted FDR",
            "next_fixture_fdr": "Next FDR",
            "next_fixture_score": "Next Fixture Score",
            "fixture_score": "3-Fixture Score",
            "fixture_count": "Fixtures Used",
            "next_fixtures": "Next Fixtures",
        }
    ).sort_values(
        by=[
            "3-Fixture Score",
            "Avg FDR",
        ],
        ascending=[
            False,
            True,
        ],
    )

    # Pandas writes the workbook; XlsxWriter handles formatting.
    with pd.ExcelWriter(
        OUTPUT_FILE,
        engine="xlsxwriter",
    ) as writer:

        workbook = writer.book

        # Formats.
        title_format = workbook.add_format(
            {
                "bold": True,
                "font_size": 16,
                "font_color": "white",
                "bg_color": "#1F4E78",
                "align": "center",
                "valign": "vcenter",
            }
        )

        header_format = workbook.add_format(
            {
                "bold": True,
                "font_color": "white",
                "bg_color": "#4472C4",
                "border": 1,
                "align": "center",
                "valign": "vcenter",
            }
        )

        core_format = workbook.add_format(
            {
                "bg_color": "#E2F0D9",
            }
        )

        watch_format = workbook.add_format(
            {
                "bg_color": "#FFF2CC",
            }
        )

        avoid_format = workbook.add_format(
            {
                "bg_color": "#F4CCCC",
            }
        )

        captain_format = workbook.add_format(
            {
                "bold": True,
                "bg_color": "#FFD966",
            }
        )

        vice_format = workbook.add_format(
            {
                "bold": True,
                "bg_color": "#D9EAD3",
            }
        )

        money_format = workbook.add_format(
            {
                "num_format": '£0.0"m"',
            }
        )

        score_format = workbook.add_format(
            {
                "num_format": "0.00",
            }
        )

        # Sheet data.
        summary.to_excel(
            writer,
            sheet_name="Summary",
            index=False,
            startrow=2,
        )

        wildcard_output.to_excel(
            writer,
            sheet_name="Wildcard Squad",
            index=False,
            startrow=2,
        )

        starting_xi.to_excel(
            writer,
            sheet_name="Starting XI",
            index=False,
            startrow=2,
        )

        bench.to_excel(
            writer,
            sheet_name="Bench",
            index=False,
            startrow=2,
        )

        top_goalkeepers.to_excel(
            writer,
            sheet_name="Top Goalkeepers",
            index=False,
            startrow=2,
        )

        top_defenders.to_excel(
            writer,
            sheet_name="Top Defenders",
            index=False,
            startrow=2,
        )

        top_midfielders.to_excel(
            writer,
            sheet_name="Top Midfielders",
            index=False,
            startrow=2,
        )

        top_forwards.to_excel(
            writer,
            sheet_name="Top Forwards",
            index=False,
            startrow=2,
        )

        all_players.to_excel(
            writer,
            sheet_name="All Players",
            index=False,
            startrow=2,
        )

        fixture_output.to_excel(
            writer,
            sheet_name="Fixtures",
            index=False,
            startrow=2,
        )

        weight_history.to_excel(
            writer,
            sheet_name="Weight History",
            index=False,
            startrow=2,
        )

        weight_summary.to_excel(
            writer,
            sheet_name="Weight Summary",
            index=False,
            startrow=2,
        )

        next_gw_weights.to_excel(
            writer,
            sheet_name="Next GW Weights",
            index=False,
            startrow=2,
        )

        sheet_data = {
            "Summary": (
                summary,
                "FPL MODEL SUMMARY",
            ),
            "Wildcard Squad": (
                wildcard_output,
                "OPTIMIZED WILDCARD SQUAD",
            ),
            "Starting XI": (
                starting_xi,
                "OPTIMIZED STARTING XI",
            ),
            "Bench": (
                bench,
                "OPTIMIZED BENCH",
            ),
            "Top Goalkeepers": (
                top_goalkeepers,
                "TOP GOALKEEPERS",
            ),
            "Top Defenders": (
                top_defenders,
                "TOP DEFENDERS",
            ),
            "Top Midfielders": (
                top_midfielders,
                "TOP MIDFIELDERS",
            ),
            "Top Forwards": (
                top_forwards,
                "TOP FORWARDS",
            ),
            "All Players": (
                all_players,
                "ALL ELIGIBLE PLAYERS",
            ),
            "Fixtures": (
                fixture_output,
                f"NEXT {UPCOMING_FIXTURE_COUNT} FIXTURE OUTLOOK",
            ),
            "Weight History": (
                weight_history,
                "GAMEWEEK-BY-GAMEWEEK LEARNED WEIGHTS",
            ),
            "Weight Summary": (
                weight_summary,
                "RUNNING MEAN + MEDIAN WEIGHTS",
            ),
            "Next GW Weights": (
                next_gw_weights,
                "NEXT GAMEWEEK ESTIMATE = RUNNING MEDIAN",
            ),
        }

        for (
            sheet_name,
            (
                dataframe,
                title,
            ),
        ) in sheet_data.items():

            worksheet = writer.sheets[
                sheet_name
            ]

            max_col = max(
                len(
                    dataframe.columns
                )
                - 1,
                0,
            )

            worksheet.merge_range(
                0,
                0,
                0,
                max_col,
                title,
                title_format,
            )

            worksheet.set_row(
                0,
                24,
            )

            # Apply custom headers over pandas defaults.
            for col_num, value in enumerate(
                dataframe.columns.values
            ):
                worksheet.write(
                    2,
                    col_num,
                    value,
                    header_format,
                )

            worksheet.freeze_panes(
                3,
                0,
            )

            worksheet.autofilter(
                2,
                0,
                2 + len(dataframe),
                max_col,
            )

            # Sensible widths.
            for col_num, column in enumerate(
                dataframe.columns
            ):
                sample_values = (
                    dataframe[column]
                    .astype(str)
                    .head(100)
                    .tolist()
                )

                max_length = max(
                    [
                        len(str(column))
                    ]
                    + [
                        len(str(value))
                        for value
                        in sample_values
                    ]
                    + [8]
                )

                width = min(
                    max_length + 2,
                    42,
                )

                if column == "Next Fixtures":
                    width = 42

                if column in [
                    "Player",
                    "Team",
                ]:
                    width = max(
                        width,
                        14,
                    )

                worksheet.set_column(
                    col_num,
                    col_num,
                    width,
                )

            # Number formats.
            if "Price" in dataframe.columns:
                idx = dataframe.columns.get_loc(
                    "Price"
                )

                worksheet.set_column(
                    idx,
                    idx,
                    10,
                    money_format,
                )

            score_columns = [
                "Final Score",
                "Retention",
                "Captain Score",
                "PPG",
                "Form",
                "Pts/£m",
                "Avg FDR",
                "Weighted FDR",
                "Next FDR",
                "Next Fixture Score",
                "3-Fixture Score",
                "Team Context",
                "Team Attack",
                "Team Defence",
                "xGI",
                "xGI/90",
            ]

            for column in score_columns:
                if column in dataframe.columns:
                    idx = dataframe.columns.get_loc(
                        column
                    )

                    worksheet.set_column(
                        idx,
                        idx,
                        12,
                        score_format,
                    )

            # Segment conditional formatting.
            if (
                "Segment"
                in dataframe.columns
                and len(dataframe) > 0
            ):
                segment_col = dataframe.columns.get_loc(
                    "Segment"
                )

                first_row = 3
                last_row = 2 + len(
                    dataframe
                )

                worksheet.conditional_format(
                    first_row,
                    segment_col,
                    last_row,
                    segment_col,
                    {
                        "type": "text",
                        "criteria": "containing",
                        "value": "Core",
                        "format": core_format,
                    },
                )

                worksheet.conditional_format(
                    first_row,
                    segment_col,
                    last_row,
                    segment_col,
                    {
                        "type": "text",
                        "criteria": "containing",
                        "value": "Watch",
                        "format": watch_format,
                    },
                )

                worksheet.conditional_format(
                    first_row,
                    segment_col,
                    last_row,
                    segment_col,
                    {
                        "type": "text",
                        "criteria": "containing",
                        "value": "Avoid",
                        "format": avoid_format,
                    },
                )

            # Captain / vice highlighting.
            if (
                "Role"
                in dataframe.columns
                and len(dataframe) > 0
            ):
                role_col = dataframe.columns.get_loc(
                    "Role"
                )

                first_row = 3
                last_row = 2 + len(
                    dataframe
                )

                worksheet.conditional_format(
                    first_row,
                    role_col,
                    last_row,
                    role_col,
                    {
                        "type": "text",
                        "criteria": "containing",
                        "value": "CAPTAIN",
                        "format": captain_format,
                    },
                )

                worksheet.conditional_format(
                    first_row,
                    role_col,
                    last_row,
                    role_col,
                    {
                        "type": "text",
                        "criteria": "containing",
                        "value": "VICE",
                        "format": vice_format,
                    },
                )

    print(
        f"Excel output created: "
        f"{OUTPUT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print(
        "Fetching current Fantasy "
        "Premier League data..."
    )

    (
        players_df,
        teams_df,
        fixtures_df,
    ) = get_fpl_data()

    players = prepare_players(
        players_df,
        teams_df,
    )

    players = add_team_context(
        players,
        teams_df,
    )

    (
        players,
        fdr,
    ) = add_fixture_strength(
        players,
        fixtures_df,
        teams_df,
    )

    players = (
        add_retention_and_captain_scores(
            players
        )
    )

    players = add_player_segments(
        players
    )

    players = add_final_score(
        players
    )

    # --------------------------------------------------------
    # SAFE GAMEWEEK LEARNING
    # --------------------------------------------------------
    # First learn any completed GW for which we already have a
    # genuine PRE-GW snapshot.
    if AUTO_LEARN_FROM_SAVED_SNAPSHOTS:
        weight_history = (
            auto_learn_from_saved_snapshots(
                fixtures_df
            )
        )
    else:
        weight_history = load_weight_history()

    (
        weight_summary,
        next_gw_weights,
    ) = summarize_weight_history(
        weight_history
    )

    players = apply_learned_median_scores(
        players,
        weight_history,
    )

    # Save the upcoming GW snapshot only AFTER all current model
    # features have been calculated. Existing snapshots are protected.
    if AUTO_SAVE_NEXT_GW_SNAPSHOT:
        next_gameweek = get_next_unfinished_gameweek(
            fixtures_df
        )

        if next_gameweek is not None:
            save_pre_gameweek_snapshot(
                players,
                gameweek=next_gameweek,
                events_df=events_df,
                refresh_before_deadline=True,
            )

    top_goalkeepers = build_top_players(
        players,
        "goalkeeper",
        TOP_COUNTS[
            "goalkeeper"
        ],
    )

    top_defenders = build_top_players(
        players,
        "defender",
        TOP_COUNTS[
            "defender"
        ],
    )

    top_midfielders = build_top_players(
        players,
        "midfielder",
        TOP_COUNTS[
            "midfielder"
        ],
    )

    top_forwards = build_top_players(
        players,
        "forward",
        TOP_COUNTS[
            "forward"
        ],
    )

    (
        wildcard_squad,
        total_cost,
        starter_score,
        bench_score,
        bench_cost,
        captain_bonus,
        objective_score,
    ) = optimize_wildcard(
        players
    )

    write_excel_output(
        players,
        fdr,
        top_goalkeepers,
        top_defenders,
        top_midfielders,
        top_forwards,
        wildcard_squad,
        total_cost,
        starter_score,
        bench_score,
        bench_cost,
        captain_bonus,
        objective_score,
        weight_history,
        weight_summary,
        next_gw_weights,
    )


if __name__ == "__main__":
    main()




# ============================================================
# SAFE LEARNING WORKFLOW
# ============================================================
#
# NORMAL RUN BEFORE A DEADLINE
# ----------------------------
# The script automatically saves:
#
#     fpl_snapshots/pre_gwN_snapshot.csv
#
# for the next unfinished Gameweek, if the snapshot does not already exist.
#
# IMPORTANT:
# Existing snapshots are NEVER overwritten automatically.
#
#
# NORMAL RUN AFTER THE GAMEWEEK FINISHES
# --------------------------------------
# The script detects that GW as completed.
#
# If pre_gwN_snapshot.csv exists and GW N is not already learned:
#
#     saved PRE-GW inputs
#         +
#     actual GW N player points
#         ↓
#     fit GW N weights
#         ↓
#     append/update fpl_gameweek_weight_history.csv
#
#
# RUNNING MODEL
# -------------
# For every feature:
#
#     Mean Weight
#     Median Weight
#     Next GW Estimate = Median Weight
#
#
# GW1 NOTE
# --------
# If no genuine PRE-GW1 snapshot exists, this script intentionally does
# NOT learn GW1 from today's post-GW1 bootstrap data.
#
# GW1 can only be added later if we reconstruct a genuine pre-season/
# pre-GW1 dataset from historical sources.
#


# ============================================================
# AUTONOMOUS DEPLOYMENT
# ============================================================
#
# Run this file on a schedule (recommended: hourly).
#
# Every scheduled run is idempotent:
#
# 1. Detect completed GWs.
# 2. Learn any completed GW that has a valid saved pre-GW snapshot and has
#    not already been learned.
# 3. Detect the next unfinished GW.
# 4. If its deadline has NOT passed, refresh that GW's snapshot.
# 5. If its deadline HAS passed, never touch the snapshot again.
#
# Why refresh before deadline?
# ----------------------------
# If the job runs hourly, an early-week snapshot is useful as a backup, but
# the final scheduled run before deadline automatically replaces it with the
# freshest legal PRE-GW data. The user does not need to remember anything.
#
# fpl_automation_state.csv records snapshot/learning actions for auditing.
#
