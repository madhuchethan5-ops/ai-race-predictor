import os
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_extras.grid import grid
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    layout="wide",
    page_title="AI Race Master Pro",
    page_icon="🏎️"
)

# ---------------------------------------------------------
# ICON PATHS
# ---------------------------------------------------------

TERRAIN_ICONS = {
    "Expressway": "assets/terrain/Expressway.png",
    "Highway": "assets/terrain/Highway.png",
    "Bumpy": "assets/terrain/Bumpy.png",
    "Dirt": "assets/terrain/Dirt.png",
    "Potholes": "assets/terrain/Potholes.png",
    "Desert": "assets/terrain/Desert.png",
}

VEHICLE_ICONS = {
    "Supercar": "assets/vehicles/Supercar.png",
    "Sports Car": "assets/vehicles/SportsCar.png",
    "Car": "assets/vehicles/Car.png",
    "Stock Car": "assets/vehicles/StockCar.png",
    "SUV": "assets/vehicles/SUV.png",
    "ORV": "assets/vehicles/ORV.png",
    "ATV": "assets/vehicles/ATV.png",
    "Motorcycle": "assets/vehicles/Motorcycle.png",
    "Monster Truck": "assets/vehicles/MonsterTruck.png",
}

def clickable_tile(label, img_path, selected=False, disabled=False, key="tile"):
    border_color = "#E53935" if selected else "#CCCCCC"
    opacity = "0.4" if disabled else "1.0"
    pointer = "not-allowed" if disabled else "pointer"

    tile_html = f"""
    <style>
    .tile-container-{key} {{
        border: 3px solid {border_color};
        border-radius: 10px;
        padding: 6px;
        text-align: center;
        cursor: {pointer};
        opacity: {opacity};
        transition: 0.15s ease-in-out;
    }}
    .tile-container-{key}:hover {{
        transform: scale(1.03);
    }}
    </style>

    <div class="tile-container-{key}" onclick="document.getElementById('{key}').click()">
        <img src="{img_path}" style="width:100%; height:auto; border-radius:6px;" />
        <div style="margin-top:6px; font-size:0.85rem;">{label}</div>
    </div>

    <input type="checkbox" id="{key}" style="display:none;" />
    """

    return tile_html
# =========================================================
# HIDDEN LAP STATS + ESTIMATOR
# =========================================================
from collections import Counter, defaultdict
import numpy as np

def build_hidden_lap_stats(history: pd.DataFrame):
    stats = {
        "global": {1: Counter(), 2: Counter(), 3: Counter()},
        "conditional": defaultdict(lambda: {1: Counter(), 2: Counter(), 3: Counter()}),
        "length": {1: [], 2: [], 3: []},
    }

    if history is None or history.empty:
        return stats

    lane_to_idx = {"Lap 1": 1, "Lap 2": 2, "Lap 3": 3}

    for _, row in history.iterrows():
        winner = row.get("Actual_Winner")
        if pd.isna(winner):
            continue

        lap_tracks = {
            1: row.get("Lap_1_Track"),
            2: row.get("Lap_2_Track"),
            3: row.get("Lap_3_Track"),
        }
        lap_lens = {
            1: row.get("Lap_1_Len"),
            2: row.get("Lap_2_Len"),
            3: row.get("Lap_3_Len"),
        }

        # global stats
        for k in (1, 2, 3):
            t = lap_tracks[k]
            L = lap_lens[k]
            if pd.isna(t) or pd.isna(L):
                continue
            stats["global"][k][t] += 1
            stats["length"][k].append(L)

        # conditional stats
        revealed_idx = lane_to_idx.get(row.get("Lane"))
        if revealed_idx in (1, 2, 3):
            revealed_track = lap_tracks[revealed_idx]
            key = (winner, revealed_idx, revealed_track)
            for k in (1, 2, 3):
                t = lap_tracks[k]
                if pd.isna(t):
                    continue
                stats["conditional"][key][k][t] += 1

    # convert lengths to means
    for k in (1, 2, 3):
        if stats["length"][k]:
            stats["length"][k] = float(np.mean(stats["length"][k]))
        else:
            stats["length"][k] = 33.3 if k != 3 else 34.0

    return stats


def estimate_hidden_laps(ctx, stats, track_options, alpha: float = 0.7):
    """
    Estimate hidden lap track distributions and expected lengths.

    alpha: how much weight to give to history vs uniform prior.
           0.7 = mostly data-driven but not allowed to collapse to a single terrain.
    """
    lap_guess = {}
    revealed_idx = ctx["idx"] + 1
    revealed_track = ctx["t"]

    # Build list of conditional keys that match the revealed lap/track
    matching_keys = [
        k for k in stats["conditional"].keys()
        if k[1] == revealed_idx and k[2] == revealed_track
    ]

    n_tracks = len(track_options)
    uniform_prior = {t: 1.0 / n_tracks for t in track_options}

    for k in (1, 2, 3):
        track_counts = Counter()

        # 1) Use conditional stats if available
        for key in matching_keys:
            track_counts.update(stats["conditional"][key][k])

        # 2) Fallback to global if conditional empty
        if not track_counts:
            track_counts = stats["global"][k].copy()

        # 3) Convert counts to probabilities
        if not track_counts:
            # No data at all → pure uniform
            data_probs = {t: 1.0 / n_tracks for t in track_options}
        else:
            total = sum(track_counts.values())
            data_probs = {t: track_counts.get(t, 0) / total for t in track_options}

        # 4) Blend with uniform prior to avoid Desert domination
        #    p_final = alpha * data_probs + (1 - alpha) * uniform
        probs = {
            t: alpha * data_probs[t] + (1.0 - alpha) * uniform_prior[t]
            for t in track_options
        }

        # 5) Renormalize to be safe
        s = sum(probs.values())
        if s > 0:
            probs = {t: p / s for t, p in probs.items()}
        else:
            probs = uniform_prior.copy()

        expected_len = stats["length"][k]

        lap_guess[k] = {
            "track_probs": probs,
            "expected_len": expected_len,
        }

    return lap_guess
# =========================================================
# TERRAIN–VEHICLE INTERACTION MATRIX
# =========================================================

def build_tv_matrix(history: pd.DataFrame):
    """
    Build a simple terrain–vehicle win-rate matrix from history.
    Returns:
        tv_matrix[(vehicle, terrain)] = win_rate (0–1) based on past races.
    """
    tv_counts = {}   # (vehicle, terrain) -> {"wins": x, "total": y}

    if history is None or history.empty:
        return {}

    # We assume history has: Vehicle_1/2/3, Lap_1_Track/2/3, Actual_Winner
    for _, row in history.iterrows():
        actual_winner = row.get("Actual_Winner")
        if pd.isna(actual_winner):
            continue

        vehicles = [
            row.get("Vehicle_1"),
            row.get("Vehicle_2"),
            row.get("Vehicle_3"),
        ]

        lap_tracks = [
            row.get("Lap_1_Track"),
            row.get("Lap_2_Track"),
            row.get("Lap_3_Track"),
        ]

        # For each vehicle and each terrain in this race, update stats
        for v in vehicles:
            if not isinstance(v, str):
                continue

            for t in lap_tracks:
                if not isinstance(t, str):
                    continue

                key = (v, t)
                if key not in tv_counts:
                    tv_counts[key] = {"wins": 0, "total": 0}

                tv_counts[key]["total"] += 1
                if v == actual_winner:
                    tv_counts[key]["wins"] += 1

    # Convert to win rates
    tv_matrix = {}
    for key, stats in tv_counts.items():
        wins = stats["wins"]
        total = max(stats["total"], 1)
        tv_matrix[key] = wins / total

    return tv_matrix


def apply_tv_adjustment(final_probs: dict, ctx: dict, tv_matrix: dict, k_type: str,
                        strength_alpha: float = 0.15):
    """
    Adjust final probabilities slightly based on terrain–vehicle strengths.

    strength_alpha: how strongly to apply the terrain–vehicle adjustment.
                    0.0 = no effect, 0.15 = gentle nudge, 0.3 = strong.
    """
    vehicles = ctx["v"]

    # 1) Extract raw strengths for this terrain
    strengths = {}
    for v in vehicles:
        key = (v, k_type)
        strengths[v] = tv_matrix.get(key, 0.5)  # 0.5 = neutral if no data

    # 2) Normalize strengths to sum to 1.0 (relative tendency)
    total_strength = sum(strengths.values())
    norm_strengths = {v: strengths[v] / total_strength for v in vehicles} if total_strength > 0 else {v: 1.0 / len(vehicles) for v in vehicles}
    if avg_strength <= 0:
        return final_probs, strengths  # safety

    norm_strengths = {v: strengths[v] / avg_strength for v in vehicles}

    # 3) Apply a small multiplicative adjustment to probabilities
    #    adjusted_p = p * (1 + alpha*(s - 1))
    adjusted = {}
    for v in vehicles:
        base_p = final_probs[v]
        s = norm_strengths[v]
        factor = 1.0 + strength_alpha * (s - 1.0)
        adjusted[v] = base_p * factor

    # 4) Renormalize to keep sum around the same scale
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {v: (adjusted[v] / total) * sum(final_probs.values())
                    for v in vehicles}
    else:
        adjusted = final_probs.copy()

    return adjusted, strengths
# ---------------------------------------------------------
# CONFIDENCE BAR
# ---------------------------------------------------------

def get_confidence_color(prob: float) -> str:
    if prob >= 70:
        return "#2e7d32"
    elif prob >= 40:
        return "#f9a825"
    else:
        return "#c62828"

def confidence_bar(vehicle: str, prob: float):
    color = get_confidence_color(prob)
    st.markdown(
        f"""
        <div style="margin-bottom:8px;">
            <div style="font-weight:600; margin-bottom:2px;">{vehicle}</div>
            <div style="background:#eee; height:10px; border-radius:5px;">
                <div style="
                    width:{prob}%;
                    height:10px;
                    background:{color};
                    border-radius:5px;">
                </div>
            </div>
            <div style="font-size:12px; color:#555; margin-top:2px;">
                {prob:.1f}% confidence
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------------------------------
# SPEED DATA + CONSTANTS
# ---------------------------------------------------------

SPEED_DATA = {
    "Monster Truck": {"Expressway": 110, "Desert": 55, "Dirt": 81, "Potholes": 48, "Bumpy": 75, "Highway": 100},
    "ORV":           {"Expressway": 140, "Desert": 57, "Dirt": 92, "Potholes": 49, "Bumpy": 76, "Highway": 112},
    "Motorcycle":    {"Expressway": 94,  "Desert": 45, "Dirt": 76, "Potholes": 36, "Bumpy": 66, "Highway": 89},
    "Stock Car":     {"Expressway": 100, "Desert": 50, "Dirt": 80, "Potholes": 45, "Bumpy": 72, "Highway": 99},
    "SUV":           {"Expressway": 180, "Desert": 63, "Dirt": 100, "Pothholes": 60, "Bumpy": 80, "Highway": 143},
    "Car":           {"Expressway": 235, "Desert": 70, "Dirt": 120, "Pothholes": 68, "Bumpy": 81, "Highway": 180},
    "ATV":           {"Expressway": 80,  "Desert": 40, "Dirt": 66, "Pothholes": 32, "Bumpy": 60, "Highway": 80},
    "Sports Car":    {"Expressway": 300, "Desert": 72, "Dirt": 130, "Pothholes": 72, "Bumpy": 91, "Highway": 240},
    "Supercar":      {"Expressway": 390, "Desert": 80, "Dirt": 134, "Pothholes": 77, "Bumpy": 99, "Highway": 320},
}

ALL_VEHICLES = sorted(list(SPEED_DATA.keys()))
TRACK_OPTIONS = sorted(list(SPEED_DATA["Car"].keys()))
VALID_TRACKS = set(TRACK_OPTIONS)

HISTORY_FILE = "race_history.csv"

TRACK_ALIASES = {
    "Road": "Highway",
    "road": "Highway",
    "Normal road": "Highway",
    "normal road": "Highway",
    "Normal": "Highway",
}

# ---------------------------------------------------------
# AUTO CLEANER
# ---------------------------------------------------------

def auto_clean_history(df: pd.DataFrame):
    if df.empty:
        return df, []

    issues = []
    df = df.copy()

    lane_map = {"1": "Lap 1", "2": "Lap 2", "3": "Lap 3", 1: "Lap 1", 2: "Lap 2", 3: "Lap 3"}
    if "Lane" in df.columns:
        df["Lane"] = df["Lane"].replace(lane_map)

    for lap in [1, 2, 3]:
        col = f"Lap_{lap}_Track"
        if col not in df.columns:
            continue
        df[col] = df[col].replace(TRACK_ALIASES)
        invalid_mask = ~df[col].isin(VALID_TRACKS) & df[col].notna()
        if invalid_mask.any():
            bad_vals = df.loc[invalid_mask, col].unique().tolist()
            issues.append(f"Invalid track names in {col}: {bad_vals}")
            if df[col].isin(VALID_TRACKS).any():
                most_common = df.loc[df[col].isin(VALID_TRACKS), col].mode()[0]
            else:
                most_common = list(VALID_TRACKS)[0]
            df.loc[invalid_mask, col] = most_common

    for lap in [1, 2, 3]:
        col = f"Lap_{lap}_Len"
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if all(f"Lap_{i}_Len" in df.columns for i in [1, 2, 3]):
        total = df["Lap_1_Len"] + df["Lap_2_Len"] + df["Lap_3_Len"]
        bad_len_mask = total.notna() & (total != 100)
        if bad_len_mask.any():
            issues.append("Some rows have Lap lengths not summing to 100. Normalizing these rows.")
            total_bad = total[bad_len_mask]
            df.loc[bad_len_mask, ["Lap_1_Len", "Lap_2_Len", "Lap_3_Len"]] = (
                df.loc[bad_len_mask, ["Lap_1_Len", "Lap_2_Len", "Lap_3_Len"]]
                .div(total_bad, axis=0)
                * 100
            )

    return df, issues

# ---------------------------------------------------------
# HISTORY LOAD / SAVE
# ---------------------------------------------------------

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(HISTORY_FILE, encoding="utf-8", engine="python")
            df.columns = [c.replace("\ufeff", "") for c in df.columns]
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        except Exception:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    required_cols = [
        'Vehicle_1','Vehicle_2','Vehicle_3',
        'Lap_1_Track','Lap_1_Len',
        'Lap_2_Track','Lap_2_Len',
        'Lap_3_Track','Lap_3_Len',
        'Actual_Winner','Predicted_Winner',
        'Lane','Top_Prob','Was_Correct',
        'Sim_Predicted_Winner','ML_Predicted_Winner',
        'Sim_Top_Prob','ML_Top_Prob',
        'Sim_Was_Correct','ML_Was_Correct',
        'Timestamp'
    ]
    for c in required_cols:
        if c not in df.columns:
            df[c] = np.nan

    df = df.replace("None", np.nan)
    df, issues = auto_clean_history(df)
    st.session_state["data_quality_issues"] = issues

    numeric_cols = [
        "Top_Prob", "Was_Correct",
        "Sim_Top_Prob","ML_Top_Prob",
        "Sim_Was_Correct","ML_Was_Correct"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def save_history(df: pd.DataFrame):
    df.to_csv(HISTORY_FILE, index=False)

def add_race_result(history_df: pd.DataFrame, row_dict: dict):
    history_df.loc[len(history_df)] = row_dict
    return history_df

history = load_history()

# ---------------------------------------------------------
# 4. ML FEATURE ENGINEERING (LEAK-SAFE) + TRAINING
# ---------------------------------------------------------

def add_leakage_safe_win_rates(df: pd.DataFrame) -> pd.DataFrame:
    if "Timestamp" not in df.columns:
        df = df.copy()
        df["Timestamp"] = pd.Timestamp.now()

    df = df.sort_values("Timestamp").reset_index(drop=True)
    win_counts = {}
    race_counts = {}

    v1_rates, v2_rates, v3_rates = [], [], []

    for _, row in df.iterrows():
        def rate(v):
            rc = race_counts.get(v, 0)
            wc = win_counts.get(v, 0)
            if rc > 0:
                return wc / rc
            return 0.33

        v1_rates.append(rate(row["Vehicle_1"]))
        v2_rates.append(rate(row["Vehicle_2"]))
        v3_rates.append(rate(row["Vehicle_3"]))

        for v in [row["Vehicle_1"], row["Vehicle_2"], row["Vehicle_3"]]:
            if pd.notna(v):
                race_counts[v] = race_counts.get(v, 0) + 1
        w = row["Actual_Winner"]
        if pd.notna(w):
            win_counts[w] = win_counts.get(w, 0) + 1

    df["V1_win_rate"] = v1_rates
    df["V2_win_rate"] = v2_rates
    df["V3_win_rate"] = v3_rates
    return df


def build_training_data(history_df: pd.DataFrame):
    df = history_df.copy()

    df = df.dropna(subset=[
        "Actual_Winner",
        "Vehicle_1", "Vehicle_2", "Vehicle_3",
        "Lap_1_Track", "Lap_2_Track", "Lap_3_Track",
        "Lap_1_Len", "Lap_2_Len", "Lap_3_Len",
        "Lane",
        "Timestamp"
    ])
    if df.empty:
        return None, None, None

    def winner_index(row):
        vs = [row["Vehicle_1"], row["Vehicle_2"], row["Vehicle_3"]]
        if row["Actual_Winner"] not in vs:
            return None
        return vs.index(row["Actual_Winner"])

    df["winner_idx"] = df.apply(winner_index, axis=1)
    df = df.dropna(subset=["winner_idx"])
    if df.empty:
        return None, None, None

    def is_high_speed(track):
        return track in ["Expressway", "Highway"]

    def is_rough(track):
        return track in ["Dirt", "Bumpy", "Potholes"]

    df["high_speed_share"] = (
        df["Lap_1_Track"].apply(is_high_speed).astype(int) +
        df["Lap_2_Track"].apply(is_high_speed).astype(int) +
        df["Lap_3_Track"].apply(is_high_speed).astype(int)
    ) / 3.0

    df["rough_share"] = (
        df["Lap_1_Track"].apply(is_rough).astype(int) +
        df["Lap_2_Track"].apply(is_rough).astype(int) +
        df["Lap_3_Track"].apply(is_rough).astype(int)
    ) / 3.0

    df = add_leakage_safe_win_rates(df)

    y = df["winner_idx"].astype(int)

    feature_cols = [
        "Vehicle_1", "Vehicle_2", "Vehicle_3",
        "Lap_1_Track", "Lap_2_Track", "Lap_3_Track",
        "Lap_1_Len", "Lap_2_Len", "Lap_3_Len",
        "Lane",
        "high_speed_share", "rough_share",
        "V1_win_rate", "V2_win_rate", "V3_win_rate",
    ]

    cat_features = [
        "Vehicle_1", "Vehicle_2", "Vehicle_3",
        "Lap_1_Track", "Lap_2_Track", "Lap_3_Track",
        "Lane"
    ]

    num_features = [
        "Lap_1_Len", "Lap_2_Len", "Lap_3_Len",
        "high_speed_share", "rough_share",
        "V1_win_rate", "V2_win_rate", "V3_win_rate",
    ]

    X = df[feature_cols].copy()
    return X, y, (cat_features, num_features)


def train_ml_model(history_df: pd.DataFrame):
    df_recent = history_df.copy().tail(200)
    X, y, feat_info = build_training_data(df_recent)
    if X is None:
        return None, 0

    n_samples = len(X)
    if n_samples < 15:
        return None, n_samples

    cat_features, num_features = feat_info

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
            ("num", "passthrough", num_features),
        ]
    )

    clf = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.1
    )

    model = Pipeline(steps=[
        ("pre", preprocessor),
        ("clf", clf),
    ])

    model.fit(X, y)
    return model, n_samples


@st.cache_resource
def get_trained_model(history_df: pd.DataFrame):
    return train_ml_model(history_df)


# ---------------------------------------------------------
# 5. SINGLE-ROW FEATURE BUILDER FOR LIVE PREDICTIONS
# ---------------------------------------------------------

def build_single_feature_row(v1, v2, v3, k_idx, k_type):
    lap_tracks = ["Unknown", "Unknown", "Unknown"]
    lap_tracks[k_idx] = k_type

    lap_lens = [33.0, 33.0, 34.0]
    lane = f"Lap {k_idx + 1}"

    high_speed_share = (
        lap_tracks.count("Expressway") + lap_tracks.count("Highway")
    ) / 3.0

    rough_share = sum(
        1 for t in lap_tracks if t in ["Dirt", "Bumpy", "Potholes"]
    ) / 3.0

    data = {
        "Vehicle_1": v1,
        "Vehicle_2": v2,
        "Vehicle_3": v3,
        "Lap_1_Track": lap_tracks[0],
        "Lap_2_Track": lap_tracks[1],
        "Lap_3_Track": lap_tracks[2],
        "Lap_1_Len": lap_lens[0],
        "Lap_2_Len": lap_lens[1],
        "Lap_3_Len": lap_lens[2],
        "Lane": lane,
        "high_speed_share": float(high_speed_share),
        "rough_share": float(rough_share),
        "V1_win_rate": 0.33,
        "V2_win_rate": 0.33,
        "V3_win_rate": 0.33,
    }
    return pd.DataFrame([data])
# ---------------------------------------------------------
# 6. METRICS & MODEL SKILL
# ---------------------------------------------------------

def compute_basic_metrics(history: pd.DataFrame):
    if history.empty:
        return None
    df = history.dropna(subset=['Actual_Winner', 'Predicted_Winner'])
    if df.empty:
        return None
    acc = (df['Actual_Winner'] == df['Predicted_Winner']).mean()

    if 'Top_Prob' in df.columns and 'Was_Correct' in df.columns:
        cal_df = df.dropna(subset=['Top_Prob', 'Was_Correct'])
        if not cal_df.empty:
            mean_top_prob = cal_df['Top_Prob'].mean()
            mean_acc = cal_df['Was_Correct'].mean()
            calib_error = abs(mean_top_prob - mean_acc)
        else:
            mean_top_prob = np.nan
            calib_error = np.nan
    else:
        mean_top_prob = np.nan
        calib_error = np.nan

    if 'Top_Prob' in df.columns and 'Was_Correct' in df.columns:
        cal_df = df.dropna(subset=['Top_Prob', 'Was_Correct'])
        if not cal_df.empty:
            brier = ((cal_df['Top_Prob'] - cal_df['Was_Correct'])**2).mean()
        else:
            brier = np.nan
    else:
        brier = np.nan

    if 'Top_Prob' in df.columns and 'Was_Correct' in df.columns:
        cal_df = df.dropna(subset=['Top_Prob', 'Was_Correct'])
        if not cal_df.empty:
            eps = 1e-8
            p = np.clip(cal_df['Top_Prob'], eps, 1 - eps)
            y = cal_df['Was_Correct']
            log_loss = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
        else:
            log_loss = np.nan
    else:
        log_loss = np.nan

    return {
        'accuracy': acc,
        'mean_top_prob': mean_top_prob,
        'calib_error': calib_error,
        'brier': brier,
        'log_loss': log_loss
    }


def compute_model_skill(history: pd.DataFrame, window: int = 100):
    cols = ['Sim_Top_Prob', 'Sim_Was_Correct',
            'ML_Top_Prob', 'ML_Was_Correct']
    if not all(c in history.columns for c in cols):
        return None

    df = history.dropna(subset=cols).tail(window)
    if df.empty:
        return None

    sim_brier = ((df['Sim_Top_Prob'] - df['Sim_Was_Correct'])**2).mean()
    ml_brier = ((df['ML_Top_Prob'] - df['ML_Was_Correct'])**2).mean()

    return {
        "sim_brier": float(sim_brier),
        "ml_brier": float(ml_brier),
        "n": int(len(df))
    }


def compute_learning_curve(history: pd.DataFrame, window: int = 30):
    if history.empty:
        return None
    df = history.dropna(subset=['Actual_Winner', 'Predicted_Winner']).copy()
    if df.empty:
        return None
    df = df.reset_index(drop=True)
    df['Correct'] = (df['Actual_Winner'] == df['Predicted_Winner']).astype(float)

    if 'Top_Prob' in df.columns and 'Was_Correct' in df.columns:
        df2 = df.dropna(subset=['Top_Prob', 'Was_Correct']).copy()
        if df2.empty:
            df['Acc_Roll'] = df['Correct'].rolling(window).mean()
            df['Brier_Roll'] = np.nan
            return df
        df2['Brier'] = (df2['Top_Prob'] - df2['Was_Correct'])**2
        df2['Acc_Roll'] = df2['Was_Correct'].rolling(window).mean()
        df2['Brier_Roll'] = df2['Brier'].rolling(window).mean()
        return df2
    else:
        df['Acc_Roll'] = df['Correct'].rolling(window).mean()
        df['Brier_Roll'] = np.nan
        return df


def compute_learned_geometry(df: pd.DataFrame):
    results = []
    for lap in [1, 2, 3]:
        t_col = f"Lap_{lap}_Track"
        l_col = f"Lap_{lap}_Len"
        if t_col not in df.columns or l_col not in df.columns:
            continue
        tmp = df[[t_col, l_col]].dropna()
        tmp = tmp[tmp[t_col] != ""]
        if tmp.empty:
            continue
        grouped = tmp.groupby(t_col)[l_col].agg(['mean', 'std', 'count']).reset_index()
        grouped['Lap'] = lap
        grouped = grouped.rename(columns={t_col: 'Track'})
        results.append(grouped)
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame(columns=['Lap', 'Track', 'mean', 'std', 'count'])


def compute_transition_matrices(history: pd.DataFrame):
    mats = {}
    for i in range(1, 4):
        for j in range(1, 4):
            if i == j:
                continue
            c1 = f"Lap_{i}_Track"
            c2 = f"Lap_{j}_Track"
            if c1 in history.columns and c2 in history.columns:
                valid = history[[c1, c2]].dropna()
                if valid.empty:
                    continue
                mat = pd.crosstab(valid[c1], valid[c2], normalize='index') * 100
                mats[(i, j)] = mat
    return mats


def compute_drift(history: pd.DataFrame, split_ratio: float = 0.5):
    if history.empty:
        return None
    n = len(history)
    if n < 40:
        return None
    split = int(n * split_ratio)
    early = history.iloc[:split]
    late = history.iloc[split:]
    geom_early = compute_learned_geometry(early)
    geom_late = compute_learned_geometry(late)
    out = {'geometry': None, 'notes': ""}

    if geom_early is not None and geom_late is not None:
        merged = pd.merge(
            geom_early,
            geom_late,
            on=['Lap', 'Track'],
            suffixes=('_early', '_late')
        )
        if not merged.empty:
            merged['mean_diff'] = merged['mean_late'] - merged['mean_early']
            merged['mean_rel_change_%'] = 100 * merged['mean_diff'] / merged['mean_early'].replace(0, np.nan)
            out['geometry'] = merged.sort_values('mean_rel_change_%', ascending=False)
            out['notes'] = "Geometry drift computed between early and late halves."
    return out


def compute_volatility_from_probs(probs: dict):
    if not probs:
        return None
    items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    if len(items) < 2:
        return {'volatility': 0.0, 'ranking': items}
    top = items[0][1]
    second = items[1][1]
    return {'volatility': top - second, 'ranking': items}


# ---------------------------------------------------------
# PHYSICS BIAS CORRECTION
# ---------------------------------------------------------

def get_physics_bias(history_df: pd.DataFrame):
    if history_df.empty or len(history_df) < 20:
        return {}

    df = history_df.dropna(subset=['Actual_Winner', 'Sim_Was_Correct'])
    if df.empty:
        return {}

    bias = df.groupby('Actual_Winner')['Sim_Was_Correct'].mean().to_dict()

    corrected = {}
    for veh, score in bias.items():
        corrected[veh] = float(np.clip(1.0 + (score - 0.55) * 0.10, 0.97, 1.03))

    return corrected


# ---------------------------------------------------------
# 7. CORE SIMULATION ENGINE
# ---------------------------------------------------------

def run_simulation(
    v1, v2, v3,
    k_idx,
    k_type,
    history_df,
    iterations=5000,
    alpha_prior=1.0,
    beta_prior=1.0,
    smoothing=0.5,
    base_len_mean=33.3,
    base_len_std=15.0,
    calib_min_hist=50
):
    vehicles = [v1, v2, v3]

    # 1. BAYESIAN REINFORCEMENT
    vpi_raw = {v: 1.0 for v in vehicles}

    if not history_df.empty and 'Actual_Winner' in history_df.columns:
        winners = history_df['Actual_Winner'].dropna()
        wins = winners.value_counts()

        if any(c.startswith('Vehicle_') for c in history_df.columns):
            all_veh = pd.concat(
                [history_df[c] for c in history_df.columns if c.startswith('Vehicle_')],
                axis=0
            ).dropna()
            races = all_veh.value_counts()
        else:
            races = wins

        posterior_means = {}
        for v in vehicles:
            w = wins.get(v, 0)
            r = races.get(v, w)
            post = (w + alpha_prior) / (r + alpha_prior + beta_prior)
            posterior_means[v] = post

        mean_post = np.mean(list(posterior_means.values())) if posterior_means else 1.0
        if mean_post > 0:
            for v in vehicles:
                vpi_raw[v] = posterior_means[v] / mean_post

    vpi = {v: float(np.clip(vpi_raw[v], 0.7, 1.3)) for v in vehicles}

    # 2. GEOMETRY
    def learned_length_dist(track_type, lap_idx):
        if history_df.empty:
            return base_len_mean, base_len_std

        col_track = f"Lap_{lap_idx+1}_Track"
        col_len   = f"Lap_{lap_idx+1}_Len"

        if col_track not in history_df.columns or col_len not in history_df.columns:
            return base_len_mean, base_len_std

        df = history_df[[col_track, col_len]].dropna()
        df = df[df[col_track] == track_type]

        if len(df) < 5:
            mask_l1 = history_df["Lap_1_Track"] == track_type
            mask_l2 = history_df["Lap_2_Track"] == track_type
            mask_l3 = history_df["Lap_3_Track"] == track_type

            combined = pd.concat([
                history_df.loc[mask_l1, "Lap_1_Len"],
                history_df.loc[mask_l2, "Lap_2_Len"],
                history_df.loc[mask_l3, "Lap_3_Len"]
            ]).dropna()

            if len(combined) < 5:
                return base_len_mean, base_len_std

            return float(combined.mean()), float(max(combined.std(), 1.0))

        mu = float(df[col_len].mean())
        sigma = float(max(df[col_len].std(), 1.0))
        return mu, sigma

    # 3. MARKOV TRANSITIONS
    lap_probs = {0: None, 1: None, 2: None}
    if not history_df.empty:
        known_col = f"Lap_{k_idx + 1}_Track"
        if known_col in history_df.columns:
            matches = history_df[history_df[known_col] == k_type].tail(200)
            global_transitions = {}
            for j in range(3):
                if j == k_idx:
                    continue
                from_col = f"Lap_{k_idx + 1}_Track"
                to_col = f"Lap_{j + 1}_Track"
                if from_col in history_df.columns and to_col in history_df.columns:
                    valid = history_df[[from_col, to_col]].dropna()
                    if valid.empty:
                        continue
                    counts = valid.groupby([from_col, to_col]).size().unstack(fill_value=0)
                    if k_type in counts.index:
                        row = counts.loc[k_type]
                        arr = row.reindex(TRACK_OPTIONS, fill_value=0).astype(float)
                        arr = arr + smoothing
                        global_transitions[j] = arr / arr.sum()

            for j in range(3):
                if j == k_idx:
                    continue
                t_col = f"Lap_{j + 1}_Track"
                if t_col in matches.columns and not matches.empty:
                    counts = matches[t_col].value_counts()
                    arr = counts.reindex(TRACK_OPTIONS, fill_value=0).astype(float)
                    arr = arr + smoothing
                    probs = arr / arr.sum()
                    lap_probs[j] = probs.values
                if lap_probs[j] is None and j in global_transitions:
                    lap_probs[j] = global_transitions[j].values

    # 4. SAMPLE TERRAIN & LENGTHS
    sim_terrains = []
    sim_lengths = []
    for i in range(3):
        if i == k_idx:
            sim_terrains.append(np.full(iterations, k_type, dtype=object))
        else:
            p = lap_probs[i]
            if p is not None and np.isfinite(p).all() and p.sum() > 0:
                sim_terrains.append(np.random.choice(TRACK_OPTIONS, size=iterations, p=p))
            else:
                sim_terrains.append(np.random.choice(TRACK_OPTIONS, size=iterations))

        terrain_i = sim_terrains[-1]
        lengths_i = np.empty(iterations, dtype=float)

        for t in np.unique(terrain_i):
            mu, sigma = learned_length_dist(t, i)
            sigma = max(sigma, 1e-3)
            mask = (terrain_i == t)
            n = mask.sum()
            if n > 0:
                lengths_i[mask] = np.random.normal(mu, sigma, size=n)

        lengths_i = np.clip(lengths_i, 1.0, None)
        sim_lengths.append(lengths_i)

    len_matrix_raw = np.column_stack(sim_lengths)
    len_sums = len_matrix_raw.sum(axis=1, keepdims=True)
    len_sums[len_sums == 0] = 1.0
    len_matrix = len_matrix_raw / len_sums

    terrain_matrix = np.column_stack(sim_terrains)

    # PHYSICS BIAS APPLIED TO SPEED DATA
    bias_table = get_physics_bias(history_df)

    adjusted_speed_data = {}
    for veh in vehicles:
        mult = bias_table.get(veh, 1.0)
        adjusted_speed_data[veh] = {
            t: spd * mult for t, spd in SPEED_DATA[veh].items()
        }

    def sample_vehicle_times(vehicle, vpi_local):
        speed_map = adjusted_speed_data[vehicle]

        base_speed = np.empty_like(terrain_matrix, dtype=float)
        for t, spd in speed_map.items():
            mask = (terrain_matrix == t)
            base_speed[mask] = spd
        base_speed = np.clip(base_speed, 0.1, None)

        veh_factor = np.random.normal(1.0, 0.03, size=(iterations, 1))
        lap_factor = np.random.normal(1.0, 0.02, size=(iterations, 3))

        effective_speed = base_speed * veh_factor * lap_factor
        effective_speed = np.clip(effective_speed, 0.1, None)

        return np.sum(len_matrix / (effective_speed * vpi_local[vehicle]), axis=1)

    results = {v: sample_vehicle_times(v, vpi) for v in vehicles}

    # 6. RAW WIN PROBABILITIES
    total_times = np.vstack([results[v] for v in vehicles])
    winners = np.argmin(total_times, axis=0)

    freq = pd.Series(winners).value_counts(normalize=True).sort_index()
    raw_probs = np.array([freq.get(i, 0.0) for i in range(3)], dtype=float)
    raw_probs = np.clip(raw_probs, 1e-6, 1.0)
    raw_probs /= raw_probs.sum()

    # 7. TEMPERATURE CALIBRATION
    def estimate_temperature_from_history(df):
        if df.empty or 'Top_Prob' not in df.columns or 'Was_Correct' not in df.columns:
            return 1.0
        recent = df.dropna(subset=['Top_Prob', 'Was_Correct']).tail(200)
        if len(recent) < calib_min_hist:
            return 1.0
        avg_conf = recent['Top_Prob'].mean()
        avg_acc = recent['Was_Correct'].mean()
        if avg_conf <= 0 or avg_acc <= 0:
            return 1.0

        calib_error = abs(avg_conf - avg_acc)
        temp = float(np.clip(1.0 + calib_error * 2.0, 0.8, 2.0))

        return float(temp)

    temp = estimate_temperature_from_history(history_df)

    logits = np.log(raw_probs)
    calibrated_logits = logits / temp
    calibrated_probs = np.exp(calibrated_logits)
    calibrated_probs /= calibrated_probs.sum()

    win_pcts = calibrated_probs * 100.0
    return {vehicles[i]: float(win_pcts[i]) for i in range(3)}, vpi
# ---------------------------------------------------------
# FULL PREDICTION ENGINE (NO UI) — FINAL CLEAN VERSION
# ---------------------------------------------------------

def run_full_prediction(v1_sel, v2_sel, v3_sel, k_idx, k_type, history):

    model_skill = compute_model_skill(history)

    # --- Simulation-based probabilities ---
    sim_probs, vpi_res = run_simulation(
        v1_sel, v2_sel, v3_sel, k_idx, k_type, history
    )

    # --- ML-based probabilities ---
    ml_probs = None
    ml_model, n_samples = get_trained_model(history)

    if ml_model is not None:
        X_curr = build_single_feature_row(v1_sel, v2_sel, v3_sel, k_idx, k_type)
        proba = ml_model.predict_proba(X_curr)[0]
        ml_probs = {
            v1_sel: float(proba[0] * 100.0),
            v2_sel: float(proba[1] * 100.0),
            v3_sel: float(proba[2] * 100.0),
        }

    final_probs = sim_probs
    p_ml_store = ml_probs

    # --- Blending weight ---
    blend_weight = 0.0

    if ml_probs is not None:
        blend_weight = 0.45

        if model_skill is not None:
            sim_brier = model_skill["sim_brier"]
            ml_brier = model_skill["ml_brier"]
            n_skill = model_skill["n"]

            if n_skill >= 30 and np.isfinite(sim_brier) and np.isfinite(ml_brier):
                improvement = (sim_brier - ml_brier) / max(sim_brier, 1e-8)

                if improvement > 0:
                    blend_weight = float(np.clip(
                        0.45 + improvement * 0.8, 0.45, 0.95
                    ))
                else:
                    degradation = abs(improvement)
                    blend_weight = float(np.clip(
                        0.45 - degradation * 0.4, 0.20, 0.45
                    ))

    blend_weight = float(np.clip(blend_weight, 0.20, 0.95))

    # --- Final blended probabilities ---
    if ml_probs is not None:
        final_probs = {
            v: blend_weight * ml_probs[v] + (1.0 - blend_weight) * sim_probs[v]
            for v in [v1_sel, v2_sel, v3_sel]
        }
    else:
        final_probs = sim_probs
    
    # --- Terrain–vehicle adjustment (gentle) ---
    tv_matrix = build_tv_matrix(history)
    ctx = {
        "v": [v1_sel, v2_sel, v3_sel],
        "idx": k_idx,
        "t": k_type,
        "slot": f"Lap {k_idx + 1}",
    }

    final_probs, tv_strengths = apply_tv_adjustment(
        final_probs, ctx, tv_matrix, k_type, strength_alpha=0.15
    )

    # --- Winner & meta calculations ---
    predicted_winner = max(final_probs, key=final_probs.get)
    p1 = final_probs[predicted_winner]

    expected_regret = p1 / 100.0

    sorted_probs = sorted(final_probs.items(), key=lambda kv: kv[1], reverse=True)
    (top_vehicle, p1_sorted), (_, p2) = sorted_probs[0], sorted_probs[1]
    vol_gap = round(p1_sorted - p2, 2)

    if vol_gap < 5:
        vol_label = "Chaos"
    elif vol_gap < 12:
        vol_label = "Shaky"
    else:
        vol_label = "Calm"

    if p1 < 40:
        bet_safety = "AVOID"
    elif vol_gap < 5:
        bet_safety = "AVOID"
    elif vol_gap < 12:
        bet_safety = "CAUTION"
    elif p1 >= 60 and vol_gap >= 15:
        bet_safety = "FAVORABLE"
    else:
        bet_safety = "CAUTION"

    # ---------------------------------------------------------
    # NEW: Hidden Lap Estimation
    # ---------------------------------------------------------
    hidden_stats = build_hidden_lap_stats(history)
    lap_guess = estimate_hidden_laps(
        {
            "v": [v1_sel, v2_sel, v3_sel],
            "idx": k_idx,
            "t": k_type,
            "slot": f"Lap {k_idx + 1}",
        },
        hidden_stats,
        TRACK_OPTIONS
    )

    # ---------------------------------------------------------
    # FINAL RESULT (NO DUPLICATE OVERWRITE)
    # ---------------------------------------------------------
    st.session_state['res'] = {
        'p': final_probs,
        'vpi': vpi_res,
        'ctx': {
            'v': [v1_sel, v2_sel, v3_sel],
            'idx': k_idx,
            't': k_type,
            'slot': f"Lap {k_idx + 1}",
        },
        'p_sim': sim_probs,
        'p_ml': p_ml_store,
        'meta': {
            'top_vehicle': top_vehicle,
            'top_prob': p1,
            'second_prob': p2,
            'volatility_gap_pp': vol_gap,
            'volatility_label': vol_label,
            'bet_safety': bet_safety,
            'expected_regret': expected_regret,
        },
        'hidden_guess': lap_guess,   # <-- stays here
        'tv_strengths': tv_strengths,
    }
# ---------------------------------------------------------
# 8. QUADRANT UI LAYOUT — AUTO-FIT DASHBOARD
# ---------------------------------------------------------

# --- Initialize State ---
if "selected_lap" not in st.session_state:
    st.session_state.selected_lap = None

if "selected_terrain" not in st.session_state:
    st.session_state.selected_terrain = None

if "selected_vehicles" not in st.session_state:
    st.session_state.selected_vehicles = []

if "trigger_prediction" not in st.session_state:
    st.session_state.trigger_prediction = False

# ---------------------------------------------------------
# QUADRANT LAYOUT (2×2, AUTO-FIT)
# ---------------------------------------------------------

top_left, top_right = st.columns(2)
bottom_left, bottom_right = st.columns(2)

Q1 = top_left.container()      # Top-left  : Race setup
Q2 = top_right.container()     # Top-right : Prediction
Q3 = bottom_left.container()   # Bottom-left: Save race
Q4 = bottom_right.container()  # Bottom-right: Diagnostics

# ---------------------------------------------------------
# Q1 — COMPACT RACE SETUP (TOP-LEFT) — FINAL FIXED VERSION
# ---------------------------------------------------------
with Q1:
    st.markdown("### 🏁 Race Setup")

    # -------------------------
    # 1. LAP & TERRAIN
    # -------------------------
    lap_col, terrain_col = st.columns([1, 1.3])

    with lap_col:
        st.caption("Lap")
        st.session_state.selected_lap = st.radio(
            label="",
            options=["Lap 1", "Lap 2", "Lap 3"],
            index=["Lap 1", "Lap 2", "Lap 3"].index(st.session_state.selected_lap)
            if st.session_state.selected_lap else 0,
            horizontal=True
        )

    with terrain_col:
        st.caption("Terrain")
        terrain = st.selectbox(
            label="",
            options=list(TERRAIN_ICONS.keys()),
            index=list(TERRAIN_ICONS.keys()).index(st.session_state.selected_terrain)
            if st.session_state.selected_terrain else 0,
        )
        st.session_state.selected_terrain = terrain

    # Terrain icon
    icon_col, _ = st.columns([1, 1])
    with icon_col:
        st.image(TERRAIN_ICONS[terrain], width=90)

    st.markdown("---")

    # -------------------------
    # 2. VEHICLE SELECTOR
    # -------------------------
    st.markdown("#### 🚗 Select up to 3 Vehicles")

    veh_keys = list(VEHICLE_ICONS.keys())
    MAX_VEHICLES = 3

    # Init state
    if "vehicle_selections" not in st.session_state:
        st.session_state.vehicle_selections = {v: False for v in veh_keys}

    if "prev_selected_vehicles" not in st.session_state:
        st.session_state.prev_selected_vehicles = []

    # Clear button
    clear_clicked = st.button("🧹 Clear Selection")
    if clear_clicked:
        for v in st.session_state.vehicle_selections:
            st.session_state.vehicle_selections[v] = False
        st.session_state.selected_vehicles = []
        st.session_state.prev_selected_vehicles = []

    # Render checkboxes
    rows = [veh_keys[i:i+3] for i in range(0, len(veh_keys), 3)]
    for row in rows:
        cols = st.columns(len(row))
        for i, v in enumerate(row):
            with cols[i]:
                checked = st.checkbox(
                    v,
                    value=st.session_state.vehicle_selections[v],
                    key=f"veh_chk_{v}",
                )
                st.session_state.vehicle_selections[v] = checked
                st.image(VEHICLE_ICONS[v], width=60)

    # Compute selection
    current_selected = [v for v, val in st.session_state.vehicle_selections.items() if val]
    prev_selected = st.session_state.prev_selected_vehicles

    # Enforce max 3
    if len(current_selected) > MAX_VEHICLES and len(current_selected) > len(prev_selected):
        newly_added = list(set(current_selected) - set(prev_selected))
        if newly_added:
            last_added = newly_added[0]
            st.session_state.vehicle_selections[last_added] = False
            current_selected = prev_selected.copy()
            st.warning("You can select up to 3 vehicles only.")

    # Update state
    st.session_state.selected_vehicles = current_selected
    st.session_state.prev_selected_vehicles = current_selected.copy()

    # Display selected vehicles
    if st.session_state.selected_vehicles:
        st.markdown("**Selected Vehicles:** " + ", ".join(st.session_state.selected_vehicles))
    else:
        st.caption("Select up to 3 vehicles to enable prediction.")

    # -------------------------
    # 3. RUN PREDICTION BUTTON
    # -------------------------
    ready = (
        st.session_state.selected_lap is not None and
        st.session_state.selected_terrain is not None and
        len(st.session_state.selected_vehicles) == 3
    )

    run_clicked = st.button(
        "🚀 RUN PREDICTION",
        disabled=not ready,
        use_container_width=True,
        key="run_prediction_main"
    )

    # -------------------------
    # 4. BUILD CONTEXT FOR Q3
    # -------------------------
    if run_clicked:
        # Convert lap label to index
        lap_map = {"Lap 1": 0, "Lap 2": 1, "Lap 3": 2}
        lap_idx = lap_map[st.session_state.selected_lap]

        # Store context EXACTLY as Q3 expects
        st.session_state.prediction_context = {
            "idx": lap_idx,                               # lap index
            "t": st.session_state.selected_terrain,       # terrain label (CRITICAL FIX)
            "slot": st.session_state.selected_lap,        # lane/lap label
            "v": st.session_state.selected_vehicles       # vehicles
        }

        st.session_state.trigger_prediction = True
        
# ---------------------------------------------------------
# Q2 — COMPACT PREDICTION PANEL (TOP-RIGHT) — FINAL VERSION
# ---------------------------------------------------------
with Q2:
    st.markdown("### 📡 Prediction & Bet Guidance")

    # Run prediction once when triggered
    if st.session_state.get("trigger_prediction", False):

        # --- Clear stale Save-form widget state BEFORE prediction ---
        for k in [
            "lap1_track", "lap2_track", "lap3_track",
            "lap1_len", "lap2_len", "lap3_len",
            "actual_winner"
        ]:
            if k in st.session_state:
                del st.session_state[k]

        # --- Build prediction context ---
        lap_map = {"Lap 1": 0, "Lap 2": 1, "Lap 3": 2}
        k_idx = lap_map[st.session_state.selected_lap]
        k_type = st.session_state.selected_terrain
        v1, v2, v3 = st.session_state.selected_vehicles

        # --- Run prediction (writes st.session_state['res']) ---
        run_full_prediction(v1, v2, v3, k_idx, k_type, history)

        # Reset trigger
        st.session_state.trigger_prediction = False

    # -----------------------------------------------------
    # DISPLAY PANEL
    # -----------------------------------------------------
    if 'res' not in st.session_state:
        st.info("Set up the race on the left and run a prediction.")
    else:
        res = st.session_state['res']
        meta = res['meta']
        probs = res['p']
        vpi = res['vpi']

        # Top row: global accuracy + winner
        top_col1, top_col2 = st.columns(2)

        with top_col1:
            if not history.empty and 'Actual_Winner' in history.columns:
                valid = history.dropna(subset=['Actual_Winner', 'Predicted_Winner'])
                if not valid.empty:
                    acc = (valid['Predicted_Winner'] == valid['Actual_Winner']).mean() * 100
                    st.metric("🎯 AI Accuracy", f"{acc:.1f}%")

        with top_col2:
            predicted_winner = max(probs, key=probs.get)
            st.metric("🏆 Predicted Winner", predicted_winner)

        # -----------------------------------------------------
        # 🤫 AI Guess for Hidden Laps (PLACED BELOW WINNER)
        # -----------------------------------------------------
        lg = res.get("hidden_guess")

        if lg:
            with st.expander("🤫 AI guess for hidden laps"):

                # Terrain emoji map
                TERRAIN_EMOJI = {
                    "Desert": "🏜️",
                    "Bumpy": "🪨",
                    "Expressway": "🛣️",
                    "Highway": "🚗",
                    "Dirt": "🌾",
                    "Potholes": "🕳️"
                }

                summary_lines = []

                for k in (1, 2, 3):
                    label = f"Lap {k}"

                    # If this is the revealed lap, show it directly
                    if k == res["ctx"]["idx"] + 1:
                        st.markdown(f"**{label} (revealed):** {res['ctx']['t']}")
                        continue

                    info = lg[k]
                    probs_k = info["track_probs"]
                    expected_len = info["expected_len"]

                    # Sort terrains by probability
                    sorted_probs = sorted(probs_k.items(), key=lambda x: x[1], reverse=True)
                    top_terrain, top_prob = sorted_probs[0]

                    emoji = TERRAIN_EMOJI.get(top_terrain, "🌍")

                    # Compact summary line
                    summary_lines.append(
                        f"**Lap {k}** → {emoji} **{top_terrain}‑heavy** (~{top_prob*100:.0f}%)"
                    )

                    # Full detail
                    top_str = ", ".join([
                        f"{TERRAIN_EMOJI.get(t, '🌍')} {t}: {p*100:.1f}%"
                        for t, p in sorted_probs[:3]
                    ])

                    st.markdown(
                        f"**{label} (hidden):** expected length ≈ {expected_len:.1f}%, "
                        f"top terrains → {top_str}"
                    )

                # Show compact summary
                st.markdown("### 🧭 Summary")
                for line in summary_lines:
                    st.markdown(f"- {line}")

        else:
            st.write("Not enough history to estimate hidden laps.")
            
        # -----------------------------------------------------
        # 🧬 Terrain–vehicle matchup (today's terrain)
        # -----------------------------------------------------
        tv_strengths = res.get("tv_strengths", {})

        if tv_strengths:
            st.markdown("#### 🧬 Terrain–vehicle matchup (win tendency)")
            terrain = res['ctx']['t']
            lines = []
            for v in res['ctx']['v']:
                s = tv_strengths.get(v, 0.33)  # now it's normalized
                if s > 0.45:
                    flavor = "favored"
                    icon = "🟢"
                elif s < 0.30:
                    flavor = "penalized"
                    icon = "🔴"
                else:
                    flavor = "neutral"
                    icon = "⚪"
                
                lines.append(f"- {icon} **{v}** on **{terrain}** → {flavor} (tendency ~{s*100:.0f}%)")
            for line in lines:
                st.markdown(line)
        else:
            st.markdown("#### 🧬 Terrain–vehicle matchup")
            st.caption("Not enough history yet to learn terrain–vehicle strengths.")

        # -----------------------------------------------------
        # Probabilities
        # -----------------------------------------------------
        st.markdown("#### 📊 Win Probabilities")
        for v in res['ctx']['v']:
            p_val = probs[v]
            boost = (vpi[v] - 1.0) * 100
            boost_str = f" (+{boost:.1f}% ML Boost)" if boost > 0 else ""
            st.markdown(f"- **{v}**: {p_val:.1f}%{boost_str}")
            confidence_bar(v, p_val)

        # Volatility + bet safety
        st.markdown("#### ⚡ Volatility & Safety")
        st.write(f"Volatility Gap: **{meta['volatility_gap_pp']} pp**")
        st.write(f"Market: **{meta['volatility_label']}**")

        safety = meta['bet_safety']
        if safety == "AVOID":
            st.error("**AVOID** — Too volatile or low-confidence.")
        elif safety == "CAUTION":
            st.warning("**CAUTION** — Edge exists but uncertainty is high.")
        else:
            st.success("**FAVORABLE** — Strong, stable edge detected.")

        # Tightness + regret
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        (_, p1), (_, p2) = sorted_probs[0], sorted_probs[1]
        margin = p1 - p2
        tightness = max(0, 100 - margin)

        c1, c2, c3 = st.columns(3)
        c1.metric("Race Tightness", f"{tightness:.1f}")
        c2.metric("Top‑2 Margin", f"{margin:.1f} pts")
        c3.metric("Expected Regret", f"{meta['expected_regret']:.2f}")

        # Diagnostics
        with st.expander("🔍 Detailed diagnostics"):
            if res.get('p_sim') and res.get('p_ml'):
                sim_winner = max(res['p_sim'], key=res['p_sim'].get)
                ml_winner = max(res['p_ml'], key=res['p_ml'].get)

                if sim_winner != ml_winner:
                    st.warning(
                        f"⚠️ **Model Divergence:** Physics → {sim_winner}, ML → {ml_winner}. "
                        "This race has higher uncertainty."
                    )
                else:
                    st.success("✅ Physics and ML agree on the winner.")

            st.markdown("**Context snapshot:**")
            st.json({
                "Revealed Lap": res['ctx']['slot'],
                "Revealed Track": res['ctx']['t'],
                "Winner": predicted_winner,
                "Probabilities": probs
            })
# ---------------------------------------------------------
# Q3 — SAVE RACE REPORT (BOTTOM-LEFT, CLEAN & WIDGET-SAFE)
# ---------------------------------------------------------
with Q3:
    st.markdown("### 📝 Save Race Report")

    # 🚫 If invalid vehicle selection, hide the form entirely
    if len(st.session_state.selected_vehicles) != 3:
        st.warning("Select exactly 3 vehicles to enable saving.")
        st.stop()

    # Check if prediction exists
    prediction_available = 'res' in st.session_state
    disabled_form = not prediction_available

    # Extract prediction context
    if prediction_available:
        res = st.session_state['res']
        ctx = res['ctx']
        predicted = res['p']
        predicted_winner = max(predicted, key=predicted.get)

        p_sim = res.get('p_sim', None)
        p_ml = res.get('p_ml', None)

        revealed_lap = ctx['idx']      # 0,1,2
        revealed_track = ctx['t']      # terrain used in prediction
        revealed_slot = ctx['slot']

        st.caption(
            f"Last prediction: **{predicted_winner}** on {revealed_slot} ({revealed_track})"
        )
    else:
        st.info("Run a prediction first to enable saving.")

    # Safe index helper
    def safe_index(value, options):
        try:
            return options.index(value)
        except Exception:
            return 0

    # -----------------------------
    # FORM BLOCK (ALWAYS RENDERED)
    # -----------------------------
    with st.expander("💾 Open save & training form", expanded=False):
        with st.form("race_report_form"):

            # Actual Winner
            winner = st.selectbox(
                "🏆 Actual Winner",
                ctx['v'] if prediction_available else [],
                index=None,
                placeholder="Select the actual winner...",
                disabled=disabled_form,
            )

            # Lap inputs
            c1, c2, c3 = st.columns(3)

            # -----------------------------
            # LAP 1
            # -----------------------------
            with c1:
                if prediction_available and revealed_lap == 0:
                    s1t = st.selectbox(
                        "Lap 1 Track",
                        TRACK_OPTIONS,
                        index=safe_index(revealed_track, TRACK_OPTIONS),
                        disabled=True,
                    )
                else:
                    s1t = st.selectbox(
                        "Lap 1 Track",
                        TRACK_OPTIONS,
                        disabled=disabled_form,
                    )
                s1l = st.number_input(
                    "Lap 1 %", 1, 100, 33, disabled=disabled_form
                )

            # -----------------------------
            # LAP 2
            # -----------------------------
            with c2:
                if prediction_available and revealed_lap == 1:
                    s2t = st.selectbox(
                        "Lap 2 Track",
                        TRACK_OPTIONS,
                        index=safe_index(revealed_track, TRACK_OPTIONS),
                        disabled=True,
                    )
                else:
                    s2t = st.selectbox(
                        "Lap 2 Track",
                        TRACK_OPTIONS,
                        disabled=disabled_form,
                    )
                s2l = st.number_input(
                    "Lap 2 %", 1, 100, 33, disabled=disabled_form
                )

            # -----------------------------
            # LAP 3
            # -----------------------------
            with c3:
                if prediction_available and revealed_lap == 2:
                    s3t = st.selectbox(
                        "Lap 3 Track",
                        TRACK_OPTIONS,
                        index=safe_index(revealed_track, TRACK_OPTIONS),
                        disabled=True,
                    )
                else:
                    s3t = st.selectbox(
                        "Lap 3 Track",
                        TRACK_OPTIONS,
                        disabled=disabled_form,
                    )
                s3l = st.number_input(
                    "Lap 3 %", 1, 100, 34, disabled=disabled_form
                )

            # Submit button
            save_clicked = st.form_submit_button("💾 Save & Train")

        # -----------------------------
        # SAVE LOGIC
        # -----------------------------
        if save_clicked:

            if not prediction_available:
                st.error("Run a prediction first.")
                st.stop()

            if winner is None:
                st.error("Please select the actual winner.")
                st.stop()

            s1l, s2l, s3l = float(s1l), float(s2l), float(s3l)
            if s1l + s2l + s3l != 100:
                st.error("Lap lengths must total 100%.")
                st.stop()

            if not s1t or not s2t or not s3t:
                st.error("All laps must have a track selected.")
                st.stop()

            st.session_state['last_train_probs'] = dict(predicted)
            # ---------------------------------------------------------
            # NEW: Hidden-lap guess error (AI learning from mistakes)
            # ---------------------------------------------------------
            def compute_hidden_guess_error(res, s1t, s2t, s3t, s1l, s2l, s3l):
                lg = res.get("hidden_guess")
                if not lg:
                    return None

                actual_tracks = {1: s1t, 2: s2t, 3: s3t}
                actual_lens = {1: s1l, 2: s2l, 3: s3l}

                track_err = {}
                len_err = {}

                for k in (1, 2, 3):
                    probs = lg[k]["track_probs"]
                    track_err[k] = 1.0 - probs.get(actual_tracks[k], 0.0)
                    len_err[k] = abs(lg[k]["expected_len"] - actual_lens[k])

                return track_err, len_err

            guess_errors = compute_hidden_guess_error(
                res, s1t, s2t, s3t, s1l, s2l, s3l
            )

            if guess_errors:
                track_err, len_err = guess_errors
            else:
                track_err = {1: None, 2: None, 3: None}
                len_err = {1: None, 2: None, 3: None}

            sim_pred_winner = max(p_sim, key=p_sim.get) if isinstance(p_sim, dict) else None
            ml_pred_winner = max(p_ml, key=p_ml.get) if isinstance(p_ml, dict) else None
            sim_top_prob = p_sim[sim_pred_winner] / 100.0 if sim_pred_winner else np.nan
            ml_top_prob = p_ml[ml_pred_winner] / 100.0 if ml_pred_winner else np.nan
            sim_correct = float(sim_pred_winner == winner) if sim_pred_winner else np.nan
            ml_correct = float(ml_pred_winner == winner) if ml_pred_winner else np.nan

            was_correct = float(predicted_winner == winner)
            p1 = predicted[predicted_winner] / 100.0
            surprise = round(1 - p1, 4) if was_correct == 1 else 1.0

            row = {
                'Vehicle_1': ctx['v'][0],
                'Vehicle_2': ctx['v'][1],
                'Vehicle_3': ctx['v'][2],
                'Lap_1_Track': s1t, 'Lap_1_Len': s1l,
                'Lap_2_Track': s2t, 'Lap_2_Len': s2l,
                'Lap_3_Track': s3t, 'Lap_3_Len': s3l,
                'Predicted_Winner': predicted_winner,
                'Actual_Winner': winner,
                'Lane': revealed_slot,
                'Top_Prob': p1,
                'Was_Correct': was_correct,
                'Surprise_Index': surprise,
                'Sim_Predicted_Winner': sim_pred_winner,
                'ML_Predicted_Winner': ml_pred_winner,
                'Sim_Top_Prob': sim_top_prob,
                'ML_Top_Prob': ml_top_prob,
                'Sim_Was_Correct': sim_correct,
                'ML_Was_Correct': ml_correct,
                'Hidden_Track_Error_L1': track_err[1],
                'Hidden_Track_Error_L2': track_err[2],
                'Hidden_Track_Error_L3': track_err[3],
                'Hidden_Len_Error_L1': len_err[1],
                'Hidden_Len_Error_L2': len_err[2],
                'Hidden_Len_Error_L3': len_err[3],
                'Timestamp': pd.Timestamp.now()
            }

            if history is None or history.empty:
                st.error("History failed to load — not saving to avoid data loss.")
            else:
                history = add_race_result(history, row)
                save_history(history)
                st.success("✅ Race saved! Model will update on next prediction.")
                st.rerun()
# ---------------------------------------------------------
# Q4 — LIGHTWEIGHT DIAGNOSTICS SUMMARY (BOTTOM-RIGHT)
# ---------------------------------------------------------
with Q4:
    st.markdown("### 📊 Dashboard & Diagnostics")

    if history is None or history.empty:
        st.info("Not enough history to compute analytics.")
    else:
        # 1. Quick health metrics
        metrics = compute_basic_metrics(history)
        col1, col2, col3 = st.columns(3)

        if metrics:
            col1.metric(
                "Global Accuracy",
                f"{metrics['accuracy']*100:.1f}%"
            )
            col2.metric(
                "Mean Top Prob",
                f"{metrics['mean_top_prob']*100:.1f}%"
                if pd.notna(metrics['mean_top_prob']) else "N/A"
            )
            col3.metric(
                "Calib Error |p̂ - acc|",
                f"{metrics['calib_error']*100:.2f}%"
                if pd.notna(metrics['calib_error']) else "N/A"
            )

        st.markdown("---")

        # 2. Recent drift snapshot (very compact)
        if 'Was_Correct' in history.columns and 'Top_Prob' in history.columns:
            df = history.dropna(subset=['Was_Correct', 'Top_Prob']).copy()
            if len(df) >= 20:
                window = min(20, len(df))
                df["Rolling_Accuracy"] = df["Was_Correct"].rolling(window).mean()
                df["Brier"] = (df["Top_Prob"] - df["Was_Correct"])**2
                df["Rolling_Brier"] = df["Brier"].rolling(window).mean()

                acc_now = df["Rolling_Accuracy"].iloc[-1]
                brier_now = df["Rolling_Brier"].iloc[-1]

                c1, c2 = st.columns(2)
                c1.metric("Recent Accuracy (20)", f"{acc_now*100:.1f}%")
                c2.metric("Recent Brier (20)", f"{brier_now:.3f}")

        st.caption("For full chaos, drift, and heatmap views, use the Analytics tabs below.")
        
# ---------------------------------------------------------
# 10. ANALYTICS TABS + WHAT‑IF SIMULATOR (FULL ORIGINAL LOGIC)
# ---------------------------------------------------------

st.markdown("## 📚 Full Analytics Suite")

if history is not None and not history.empty:

    tabs = st.tabs([
        "📊 Performance Dashboard",
        "📈 Learning Curves",
        "🎯 Calibration Analyzer",
        "🌊 Drift Detector",
        "⚡ Volatility & Importance",
        "🧠 ML Pattern Brain",
        "🚦 Lane Tracker",
        "🧹 Data Quality Checker",
        "📂 History",
        "🧪 What‑If Simulator"
    ])

    # -----------------------------------------------------
    # 1) PERFORMANCE DASHBOARD
    # -----------------------------------------------------
    with tabs[0]:
        st.write("### 📊 Performance Insights Dashboard")
        metrics = compute_basic_metrics(history)

        if not metrics:
            st.info("Not enough data yet to compute metrics.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Global Accuracy", f"{metrics['accuracy']*100:.1f}%")
            c2.metric("Mean Top Probability",
                      f"{metrics['mean_top_prob']*100:.1f}%" if pd.notna(metrics['mean_top_prob']) else "N/A")
            c3.metric("Calibration Error |p̂ - acc|",
                      f"{metrics['calib_error']*100:.2f}%" if pd.notna(metrics['calib_error']) else "N/A")

            c4, c5 = st.columns(2)
            c4.metric("Brier Score (↓ better)",
                      f"{metrics['brier']:.4f}" if pd.notna(metrics['brier']) else "N/A")
            c5.metric("Log Loss (↓ better)",
                      f"{metrics['log_loss']:.4f}" if pd.notna(metrics['log_loss']) else "N/A")

            st.caption("Calibration Error close to 0 means probabilities match reality. Brier/Log Loss lower = sharper, better-calibrated model.")

    # -----------------------------------------------------
    # 2) LEARNING CURVES
    # -----------------------------------------------------
    with tabs[1]:
        st.write("### 📈 Learning Curves")
        curve = compute_learning_curve(history, window=20)

        if curve is None or curve.empty:
            st.info("Need more races to build learning curves.")
        else:
            st.line_chart(curve[['Acc_Roll']], height=250)

            if 'Brier_Roll' in curve.columns and curve['Brier_Roll'].notna().any():
                st.line_chart(curve[['Brier_Roll']], height=250)
                st.caption("Top: rolling accuracy. Bottom: rolling Brier score (lower is better).")

    # -----------------------------------------------------
    # 3) CALIBRATION ANALYZER
    # -----------------------------------------------------
    with tabs[2]:
        st.write("### 🎯 Calibration Analyzer")

        if 'Top_Prob' in history.columns and 'Was_Correct' in history.columns:
            cal_df = history.dropna(subset=['Top_Prob', 'Was_Correct']).copy()

            if cal_df.empty:
                st.info("Not enough calibrated predictions yet.")
            else:
                cal_df['Bucket'] = (cal_df['Top_Prob'] * 10).astype(int) / 10.0

                calib_table = cal_df.groupby('Bucket').agg(
                    mean_prob=('Top_Prob', 'mean'),
                    emp_acc=('Was_Correct', 'mean'),
                    count=('Was_Correct', 'size')
                ).reset_index()

                st.write("#### Reliability Table")
                st.dataframe(calib_table.style.format({'mean_prob': '{:.2f}', 'emp_acc': '{:.2f}'}))

                st.line_chart(calib_table.set_index('Bucket')[['mean_prob', 'emp_acc']], height=300)
                st.caption("If the lines track each other closely, the AI is well-calibrated.")
        else:
            st.info("Top_Prob / Was_Correct not available yet for calibration analysis.")

    # -----------------------------------------------------
    # 4) DRIFT DETECTOR (GEOMETRY)
    # -----------------------------------------------------
    with tabs[3]:
        st.write("### 🌊 Drift Detector (Track Geometry)")

        drift = compute_drift(history)

        if not drift or drift['geometry'] is None or drift['geometry'].empty:
            st.info("Not enough history or drift not detectable yet.")
        else:
            st.write(drift['notes'])
            geom_df = drift['geometry']

            st.dataframe(
                geom_df[['Lap', 'Track', 'mean_early', 'mean_late', 'mean_rel_change_%']],
                use_container_width=True
            )

            st.caption("Large relative changes in mean length indicate environment or strategy drift.")

    # -----------------------------------------------------
    # 5) VOLATILITY & FEATURE IMPORTANCE
    # -----------------------------------------------------
    with tabs[4]:
        st.write("### ⚡ Volatility & Sensitivity")

        if 'res' in st.session_state:
            res = st.session_state['res']
            vol = compute_volatility_from_probs(res['p'])

            if vol:
                st.metric("Volatility (Top - Second)", f"{vol['volatility']:.1f} pp")

                if vol['volatility'] < 5:
                    st.warning("Highly volatile race: outcomes are very close.")
                elif vol['volatility'] < 15:
                    st.info("Moderately confident prediction.")
                else:
                    st.success("High confidence prediction.")

            st.write("#### Sensitivity: Speed Multiplier What-If")

            v_sel = st.selectbox("Vehicle to stress-test", res['ctx']['v'])
            mult = st.slider("Speed multiplier for selected vehicle", 0.8, 1.2, 1.0, 0.02)

            original_speed = SPEED_DATA[v_sel].copy()
            SPEED_DATA[v_sel] = {k: v * mult for k, v in original_speed.items()}

            probs_whatif, _ = run_simulation(
                res['ctx']['v'][0],
                res['ctx']['v'][1],
                res['ctx']['v'][2],
                res['ctx']['idx'],
                res['ctx']['t'],
                history
            )

            SPEED_DATA[v_sel] = original_speed

            c1, c2 = st.columns(2)
            with c1:
                st.write("Original probabilities")
                st.json(res['p'])
            with c2:
                st.write(f"With {v_sel} speed x{mult:.2f}")
                st.json(probs_whatif)

            st.caption("This approximates 'feature importance' by perturbation: how much probabilities move when one vehicle changes.")
        else:
            st.info("Run a prediction first to analyze volatility and sensitivity.")

    # -----------------------------------------------------
    # 6) ML PATTERN BRAIN
    # -----------------------------------------------------
    with tabs[5]:
        st.write("### 🧠 ML Pattern Brain (Track Transition Matrix)")

        if 'Lap_1_Track' in history.columns and 'Lap_2_Track' in history.columns:
            m = pd.crosstab(history['Lap_1_Track'], history['Lap_2_Track'], normalize='index') * 100
            st.dataframe(m.style.format("{:.0f}%").background_gradient(cmap="Blues", axis=1))

        mats = compute_transition_matrices(history)

        if mats:
            st.write("#### All learned transitions")
            for (i, j), mat in mats.items():
                st.write(f"Lap {i} → Lap {j}")
                st.dataframe(mat.style.format("{:.0f}%").background_gradient(cmap="Blues", axis=1))

    # -----------------------------------------------------
    # 7) LANE TRACKER
    # -----------------------------------------------------
    with tabs[6]:
        st.write("### 🚦 Win Rate by Lane Context")

        if 'Lane' in history.columns and history['Lane'].notna().any():
            lane_stats = pd.crosstab(history['Lane'], history['Actual_Winner'], normalize='index') * 100
            st.dataframe(lane_stats.style.format("{:.1f}%").background_gradient(cmap="YlOrRd", axis=1))
        else:
            st.info("Record more races to see Lane win rates.")
    # ---------------------------------------------------------
    # CSV HEALTH CHECK FUNCTION
    # ---------------------------------------------------------
    def csv_health_check(df: pd.DataFrame):
        issues = []
        if df is None or df.empty:
            issues.append("CSV is empty or failed to load.")
            return issues
    
        required_cols = [
            "Vehicle_1", "Vehicle_2", "Vehicle_3",
            "Lap_1_Track", "Lap_2_Track", "Lap_3_Track",
            "Lap_1_Len", "Lap_2_Len", "Lap_3_Len",
            "Actual_Winner", "Predicted_Winner"
        ]
    
        for col in required_cols:
            if col not in df.columns:
                issues.append(f"Missing column: {col}")
    
        if df.isnull().sum().sum() > 0:
            issues.append("CSV contains missing values.")
    
        return issues
    
    
    # ---------------------------------------------------------
    # TAB 7 — DATA QUALITY CHECKER
    # ---------------------------------------------------------
    with tabs[7]:
        st.write("### 🧹 Data Quality Checker")
    
        issues = st.session_state.get("data_quality_issues", [])
    
        if not issues:
            st.success("✅ No data quality issues detected by auto-cleaner.")
        else:
            st.warning("⚠️ Issues detected and auto-corrected at load time:")
            for i in issues:
                st.write(f"- {i}")
    
        geom = compute_learned_geometry(history)
    
        if geom is not None and not geom.empty:
            unstable = geom[geom["std"] > geom["mean"] * 0.6]
    
            if not unstable.empty:
                st.error("⚠️ Geometry instability detected:")
                st.dataframe(unstable, use_container_width=True)
            else:
                st.success("✅ Geometry looks stable.")
        else:
            st.info("Not enough data yet to evaluate geometry stability.")
    
        st.write("### 🧹 CSV Health Check")
        csv_issues = csv_health_check(history)
    
        if not csv_issues:
            st.success("✅ CSV is healthy and fully compatible.")
        else:
            st.error("⚠️ Issues detected:")
            for i in csv_issues:
                st.write(f"- {i}")

    # -----------------------------------------------------
    # 9) RAW HISTORY
    # -----------------------------------------------------
    with tabs[8]:
        st.write("### 📂 Race History")
        st.dataframe(history.sort_index(ascending=False), use_container_width=True)

        st.download_button(
            "⬇️ Download Race History",
            history.to_csv(index=False),
            "race_history.csv",
            mime="text/csv"
        )

    # -----------------------------------------------------
    # 10) WHAT‑IF SIMULATOR
    # -----------------------------------------------------
    with tabs[9]:
        st.write("### 🧪 What‑If Analysis Panel")
        st.caption("Experiment with different track reveals and vehicle combos without logging to history.")

        sim_lap_map = {"Lap 1": 0, "Lap 2": 1, "Lap 3": 2}
        sim_slot = st.selectbox("What-If: Revealed Slot", list(sim_lap_map.keys()))
        sim_idx = sim_lap_map[sim_slot]

        sim_track = st.selectbox("What-If: Revealed Track", TRACK_OPTIONS, key="whatif_track")

        c_a, c_b, c_c = st.columns(3)

        with c_a:
            sim_v1 = st.selectbox("What-If Vehicle 1", ALL_VEHICLES, index=ALL_VEHICLES.index("Supercar"), key="whatif_v1")
        with c_b:
            sim_v2 = st.selectbox("What-If Vehicle 2", [v for v in ALL_VEHICLES if v != sim_v1], key="whatif_v2")
        with c_c:
            sim_v3 = st.selectbox("What-If Vehicle 3", [v for v in ALL_VEHICLES if v not in [sim_v1, sim_v2]], key="whatif_v3")

        if st.button("Run What-If Simulation"):
            probs_sim, vpi_sim = run_simulation(sim_v1, sim_v2, sim_v3, sim_idx, sim_track, history)

            st.write("#### What-If Probabilities")
            st.json(probs_sim)

            vol_sim = compute_volatility_from_probs(probs_sim)
            if vol_sim:
                st.metric("Volatility (Top - Second)", f"{vol_sim['volatility']:.1f} pp")

            st.caption("Use this to explore setups before committing to a real predictions + telemetry cycle.")

            st.write("### 👻 Ghost Lap Scenarios")
            st.caption("Stress-test the hidden laps by assuming extreme terrain profiles.")

            probs_high, _ = run_simulation(sim_v1, sim_v2, sim_v3, sim_idx, "Expressway", history)
            probs_rough, _ = run_simulation(sim_v1, sim_v2, sim_v3, sim_idx, "Dirt", history)

            col_gh, col_gr = st.columns(2)

            with col_gh:
                st.write("Ghost High-Speed (Expressway as revealed)")
                st.json(probs_high)

            with col_gr:
                st.write("Ghost Rough (Dirt as revealed)")
                st.json(probs_rough)

            st.caption("Ghost scenarios approximate how outcomes shift if underlying laps skew high-speed vs rough.")
