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

st.set_page_config(layout="wide", page_title="AI Race Master Pro", page_icon="🏎️")
st.write("RUNNING FILE:", __file__)

# ---------------------------------------------------------
# 0. CONFIDENCE VISUALS
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
# 1. PHYSICS CONFIG
# ---------------------------------------------------------

SPEED_DATA = {
    "Monster Truck": {"Expressway": 110, "Desert": 55, "Dirt": 81, "Potholes": 48, "Bumpy": 75, "Highway": 100},
    "ORV":           {"Expressway": 140, "Desert": 57, "Dirt": 92, "Potholes": 49, "Bumpy": 76, "Highway": 112},
    "Motorcycle":    {"Expressway": 94,  "Desert": 45, "Dirt": 76, "Potholes": 36, "Bumpy": 66, "Highway": 89},
    "Stock Car":     {"Expressway": 100, "Desert": 50, "Dirt": 80, "Potholes": 45, "Bumpy": 72, "Highway": 99},
    "SUV":           {"Expressway": 180, "Desert": 63, "Dirt": 100, "Potholes": 60, "Bumpy": 80, "Highway": 143},
    "Car":           {"Expressway": 235, "Desert": 70, "Dirt": 120, "Potholes": 68, "Bumpy": 81, "Highway": 180},
    "ATV":           {"Expressway": 80,  "Desert": 40, "Dirt": 66, "Potholes": 32, "Bumpy": 60, "Highway": 80},
    "Sports Car":    {"Expressway": 300, "Desert": 72, "Dirt": 130, "Potholes": 72, "Bumpy": 91, "Highway": 240},
    "Supercar":      {"Expressway": 390, "Desert": 80, "Dirt": 134, "Potholes": 77, "Bumpy": 99, "Highway": 320},
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
# 2. AUTO CLEANER
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
# 3. HISTORY LOAD / SAVE
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
        # model-skill logging
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

    numeric_cols = ["Top_Prob", "Was_Correct",
                    "Sim_Top_Prob","ML_Top_Prob",
                    "Sim_Was_Correct","ML_Was_Correct"]
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
    """
    Compute per-vehicle win rates using only past races (expanding window),
    to avoid target leakage.
    """
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
    """
    Build supervised data: each row = race, target = Actual_Winner among 3 vehicles.
    Includes leak-safe per-vehicle win rates and terrain mix features.
    """
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
    """
    Train a gradient boosting model on recent historical races.
    Uses last 200 races, requires at least 15 samples.
    Returns (model, n_samples) or (None, 0).
    """
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
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
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
    """
    Cached ML model; invalidates automatically when history_df changes.
    """
    return train_ml_model(history_df)
# ---------------------------------------------------------
# 5. SINGLE-ROW FEATURE BUILDER FOR LIVE PREDICTIONS
# ---------------------------------------------------------

def build_single_feature_row(v1, v2, v3, k_idx, k_type):
    """
    Build a single-row DataFrame to feed the ML model for current race context.
    Unknown laps approximated as 'Unknown' with 33/33/34 lengths.
    """
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
    """
    Compare recent Brier scores for Simulation vs ML using separately logged
    top probabilities and correctness.
    """
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
    st.write("GEOMETRY FUNCTION VERSION:", "v2")
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
# PHYSICS BIAS CORRECTION (LIGHTWEIGHT)
# ---------------------------------------------------------

def get_physics_bias(history_df):
    """
    Computes how often each vehicle's simulation prediction was correct.
    If the simulation is consistently wrong for a vehicle, we adjust its speed slightly.
    """
    if history_df.empty or len(history_df) < 20:
        return {}

    df = history_df.dropna(subset=['Actual_Winner', 'Sim_Was_Correct'])
    if df.empty:
        return {}

    # Mean correctness per vehicle
    bias = df.groupby('Actual_Winner')['Sim_Was_Correct'].mean().to_dict()

    # Convert correctness into a small speed multiplier
    # If sim correctness = 0.40 → multiplier = 0.97 (slightly slower)
    # If sim correctness = 0.70 → multiplier = 1.02 (slightly faster)
    corrected = {}
    for veh, score in bias.items():
        corrected[veh] = float(np.clip(1.0 + (score - 0.55) * 0.10, 0.97, 1.03))

    return corrected

# ---------------------------------------------------------
# 7. CORE SIMULATION (BAYES + GEOMETRY + MARKOV + VECTORISED PHYSICS)
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

        # --- PHYSICS BIAS ADJUSTMENT (must come BEFORE sample_vehicle_times) ---
    bias_table = get_physics_bias(history_df)

    adjusted_speed_data = {}
    for veh in vehicles:
        mult = bias_table.get(veh, 1.0)
        adjusted_speed_data[veh] = {
            t: spd * mult for t, spd in SPEED_DATA[veh].items()
        }

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

    # 2. GEOMETRY (UPDATED)
    def learned_length_dist(track_type):
        """
        Uses actual historical lap lengths for this track type.
        Much more accurate than the flat 33.3% assumption.
        """
        if history_df.empty:
            return base_len_mean, base_len_std

        mask_l1 = (history_df["Lap_1_Track"] == track_type)
        mask_l2 = (history_df["Lap_2_Track"] == track_type)
        mask_l3 = (history_df["Lap_3_Track"] == track_type)

        combined = pd.concat([
            history_df.loc[mask_l1, "Lap_1_Len"],
            history_df.loc[mask_l2, "Lap_2_Len"],
            history_df.loc[mask_l3, "Lap_3_Len"]
        ]).dropna()

        if len(combined) < 5:
            return base_len_mean, base_len_std

        return float(combined.mean()), float(combined.std())
        
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
            mu, sigma = learned_length_dist(t)
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

    # 5. NOISE MODELING (vectorized speed lookup per track)
    def sample_vehicle_times(vehicle):
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

        return np.sum(len_matrix / (effective_speed * vpi[vehicle]), axis=1)

        # Apply physics bias (small correction)
    bias_table = get_physics_bias(history_df)

    # Adjust SPEED_DATA dynamically
    adjusted_speed_data = {}
    for veh in vehicles:
        mult = bias_table.get(veh, 1.0)
        adjusted_speed_data[veh] = {t: spd * mult for t, spd in SPEED_DATA[veh].items()}
    results = {v: sample_vehicle_times(v) for v in vehicles}

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
        ratio = avg_conf / max(avg_acc, 1e-3)
        temp = np.clip(ratio, 0.7, 1.5)
        return float(temp)

    temp = estimate_temperature_from_history(history_df)

    logits = np.log(raw_probs)
    calibrated_logits = logits / temp
    calibrated_probs = np.exp(calibrated_logits)
    calibrated_probs /= calibrated_probs.sum()

    win_pcts = calibrated_probs * 100.0
    return {vehicles[i]: float(win_pcts[i]) for i in range(3)}, vpi
    # ---------------------------------------------------------
# 8. SIDEBAR (SETUP & PREDICTION) WITH PERFORMANCE-DRIVEN BLENDING
# ---------------------------------------------------------

with st.sidebar:
    st.header("🚦 Race Setup")
    lap_map = {"Lap 1": 0, "Lap 2": 1, "Lap 3": 2}
    slot_name = st.selectbox("Revealed Slot", list(lap_map.keys()))
    k_idx = lap_map[slot_name]
    k_type = st.selectbox("Revealed Track", TRACK_OPTIONS)

    st.divider()
    v1_sel = st.selectbox("Vehicle 1", ALL_VEHICLES, index=ALL_VEHICLES.index("Supercar"))
    v2_sel = st.selectbox("Vehicle 2", [v for v in ALL_VEHICLES if v != v1_sel], index=0)
    v3_sel = st.selectbox("Vehicle 3", [v for v in ALL_VEHICLES if v not in [v1_sel, v2_sel]], index=0)

    # Precompute model skill based on history
    model_skill = compute_model_skill(history)

    if st.button("🚀 PREDICT", type="primary", use_container_width=True):
        # Simulation-based probabilities
        sim_probs, vpi_res = run_simulation(v1_sel, v2_sel, v3_sel, k_idx, k_type, history)

        # ML-based probabilities (if model exists)
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

        # Performance-aware blending
        if ml_probs is not None and model_skill is not None:
            sim_brier = model_skill["sim_brier"]
            ml_brier = model_skill["ml_brier"]
            n_skill = model_skill["n"]

            if n_skill >= 30 and np.isfinite(sim_brier) and np.isfinite(ml_brier):
                if ml_brier < sim_brier:
                    # ML sharper than sim: weight increases with relative improvement
                    improvement = (sim_brier - ml_brier) / max(sim_brier, 1e-8)
                    blend_weight = float(np.clip(0.3 + 0.5 * improvement, 0.3, 0.8))
                else:
                    # ML worse: throttle down
                    degradation = (ml_brier - sim_brier) / max(sim_brier, 1e-8)
                    blend_weight = float(np.clip(0.3 - 0.3 * degradation, 0.0, 0.3))
            else:
                blend_weight = 0.4  # not enough skill data yet
        elif ml_probs is not None:
            # Model exists but no skill history yet
            blend_weight = 0.4
        else:
            blend_weight = 0.0  # no ML

        if ml_probs is not None and blend_weight > 0.0:
            blended = {}
            for v in [v1_sel, v2_sel, v3_sel]:
                blended[v] = (
                    blend_weight * ml_probs[v] +
                    (1.0 - blend_weight) * sim_probs[v]
                )
            final_probs = blended

        st.session_state['res'] = {
            'p': final_probs,
            'vpi': vpi_res,
            'ctx': {'v': [v1_sel, v2_sel, v3_sel], 'idx': k_idx, 't': k_type, 'slot': slot_name},
            'p_sim': sim_probs,
            'p_ml': p_ml_store,
        }

# ---------------------------------------------------------
# 9. MAIN DASHBOARD
# ---------------------------------------------------------

st.title("🏁 AI RACE MASTER PRO")

if not history.empty and 'Actual_Winner' in history.columns:
    valid = history.dropna(subset=['Actual_Winner', 'Predicted_Winner'])
    if not valid.empty:
        acc = (valid['Predicted_Winner'] == valid['Actual_Winner']).mean() * 100
        st.metric("🎯 AI Prediction Accuracy", f"{acc:.1f}%")

if 'res' in st.session_state:
    res = st.session_state['res']
    m_grid = grid(3, vertical_align="center")
    for v, val in res['p'].items():
        boost = (res['vpi'][v] - 1.0) * 100
        m_grid.metric(v, f"{val:.1f}%", f"+{boost:.1f}% ML Boost" if boost > 0 else None)

    # --- MODEL DIVERGENCE ALERT ---
    sim_probs = res.get('p_sim')
    ml_probs = res.get('p_ml')

    if sim_probs and ml_probs:
        sim_winner = max(sim_probs, key=sim_probs.get)
        ml_winner = max(ml_probs, key=ml_probs.get)

        if sim_winner != ml_winner:
            st.warning(
                f"⚠️ **Model Divergence:** Physics favors **{sim_winner}**, "
                f"but ML favors **{ml_winner}**. This race has higher uncertainty."
            )
        else:
            st.success("✅ **Model Consensus:** Physics and ML agree on the likely winner.")

# ---------------------------------------------------------
# 10. SAVE RACE REPORT (LOG SIM & ML SEPARATELY)
# ---------------------------------------------------------

st.divider()
st.subheader("📝 Save Race Report")

if 'res' not in st.session_state:
    st.info("Run a prediction first.")
    st.stop()

res = st.session_state['res']
ctx = res['ctx']
predicted = res['p']
predicted_winner = max(predicted, key=predicted.get)

p_sim = res.get('p_sim', None)
p_ml = res.get('p_ml', None)

revealed_lap = ctx['idx']
revealed_track = ctx['t']
revealed_slot = ctx['slot']

with st.form("race_report_form"):
    winner = st.selectbox(
        "🏆 Actual Winner",
        ctx['v'],
        index=None,
        placeholder="Select the actual winner..."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        if revealed_lap == 0:
            s1t = st.selectbox(
                "Lap 1 Track",
                TRACK_OPTIONS,
                index=TRACK_OPTIONS.index(revealed_track),
                disabled=True
            )
            s1l = st.number_input("Lap 1 %", 1, 100, 33)
        else:
            s1t = st.selectbox("Lap 1 Track", TRACK_OPTIONS)
            s1l = st.number_input("Lap 1 %", 1, 100, 33)

    with c2:
        if revealed_lap == 1:
            s2t = st.selectbox(
                "Lap 2 Track",
                TRACK_OPTIONS,
                index=TRACK_OPTIONS.index(revealed_track),
                disabled=True
            )
            s2l = st.number_input("Lap 2 %", 1, 100, 33)
        else:
            s2t = st.selectbox("Lap 2 Track", TRACK_OPTIONS)
            s2l = st.number_input("Lap 2 %", 1, 100, 33)

    with c3:
        if revealed_lap == 2:
            s3t = st.selectbox(
                "Lap 3 Track",
                TRACK_OPTIONS,
                index=TRACK_OPTIONS.index(revealed_track),
                disabled=True
            )
            s3l = st.number_input("Lap 3 %", 1, 100, 34)
        else:
            s3t = st.selectbox("Lap 3 Track", TRACK_OPTIONS)
            s3l = st.number_input("Lap 3 %", 1, 100, 34)

    save_clicked = st.form_submit_button("💾 Save & Train")

if save_clicked:
    if winner is None:
        st.error("Please select the actual winner.")
        st.stop()

    s1l = float(s1l)
    s2l = float(s2l)
    s3l = float(s3l)

    if s1l + s2l + s3l != 100:
        st.error("Lap lengths must total 100%.")
        st.stop()

    if not s1t or not s2t or not s3t:
        st.error("All laps must have a track selected.")
        st.stop()

    st.session_state['last_train_probs'] = dict(predicted)

    # Sim / ML logging for skill computation
    sim_pred_winner = None
    ml_pred_winner = None
    sim_top_prob = np.nan
    ml_top_prob = np.nan
    sim_correct = np.nan
    ml_correct = np.nan

    if isinstance(p_sim, dict):
        sim_pred_winner = max(p_sim, key=p_sim.get)
        sim_top_prob = p_sim[sim_pred_winner] / 100.0
        sim_correct = float(sim_pred_winner == winner)

    if isinstance(p_ml, dict):
        ml_pred_winner = max(p_ml, key=p_ml.get)
        ml_top_prob = p_ml[ml_pred_winner] / 100.0
        ml_correct = float(ml_pred_winner == winner)

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
        'Top_Prob': predicted[predicted_winner] / 100.0,
        'Was_Correct': float(predicted_winner == winner),
        'Sim_Predicted_Winner': sim_pred_winner,
        'ML_Predicted_Winner': ml_pred_winner,
        'Sim_Top_Prob': sim_top_prob,
        'ML_Top_Prob': ml_top_prob,
        'Sim_Was_Correct': sim_correct,
        'ML_Was_Correct': ml_correct,
        'Timestamp': pd.Timestamp.now()
    }

    if history is None or history.empty:
        st.error("History failed to load — not saving to avoid data loss.")
        st.stop()

    history = add_race_result(history, row)
    save_history(history)

    st.success("✅ Race saved. The model cache will update on next run.")
    st.rerun()
    # ---------------------------------------------------------
# 11. PREDICTION ANALYTICS PANEL
# ---------------------------------------------------------

def csv_health_check(df: pd.DataFrame):
    issues = []

    unnamed = [c for c in df.columns if "Unnamed" in c]
    if unnamed:
        issues.append(f"Unnamed columns detected: {unnamed}")

    required = [
        'Vehicle_1', 'Vehicle_2', 'Vehicle_3',
        'Lap_1_Track', 'Lap_1_Len',
        'Lap_2_Track', 'Lap_2_Len',
        'Lap_3_Track', 'Lap_3_Len',
        'Actual_Winner', 'Predicted_Winner',
        'Lane', 'Top_Prob', 'Was_Correct'
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")

    bad_rows = df[df.isna().all(axis=1)]
    if not bad_rows.empty:
        issues.append(f"Empty/malformed rows: {len(bad_rows)}")

    valid_tracks = set(TRACK_OPTIONS)
    for lap in [1, 2, 3]:
        col = f"Lap_{lap}_Track"
        if col in df.columns:
            invalid = df[~df[col].isin(valid_tracks)][col].unique()
            invalid = [x for x in invalid if pd.notna(x)]
            if invalid:
                issues.append(f"Invalid track names in {col}: {invalid}")

    return issues

if 'res' in st.session_state:
    res = st.session_state['res']
    ctx = res['ctx']
    probs = res['p']
    vpi = res['vpi']
    predicted_winner = max(probs, key=probs.get)

    st.divider()
    st.subheader("🔍 Prediction Explanation")

    explanation = ""
    if probs[predicted_winner] > 80:
        explanation += f"- **{predicted_winner}** is significantly faster on the dominant lap.\n"
    if vpi[predicted_winner] > 1.05:
        explanation += f"- Reinforcement learning shows **{predicted_winner}** has strong historical performance.\n"

    revealed_track = ctx['t']
    if revealed_track in ["Expressway", "Highway"]:
        explanation += "- High-speed tracks strongly favor Supercar / Sports Car.\n"
    if revealed_track in ["Dirt", "Bumpy", "Potholes"]:
        explanation += "- Rough tracks often favor ORV / Monster Truck.\n"

    if explanation == "":
        explanation = "The AI selected the winner based on combined physics, lap lengths, and Monte‑Carlo simulations."

    st.info(explanation)

    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    (v1p, p1), (v2p, p2) = sorted_probs[0], sorted_probs[1]
    margin = p1 - p2
    tightness = max(0, 100 - margin)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Race Tightness Score", f"{tightness:.1f}")
    with col2:
        st.metric("Top‑2 Margin", f"{margin:.1f} pts")

    st.write("### Confidence by Vehicle")
    for v, p in probs.items():
        confidence_bar(v, p)

    if 'last_train_probs' in st.session_state:
        st.write("### 📈 Confidence Change Since Last Training")
        prev = st.session_state['last_train_probs']
        rows = []
        for v, p_now in probs.items():
            p_prev = prev.get(v, None)
            delta = p_now - p_prev if p_prev is not None else None
            rows.append({
                "Vehicle": v,
                "Now %": p_now,
                "Prev %": p_prev if p_prev is not None else "-",
                "Δ (Now - Prev)": f"{delta:+.1f}" if delta is not None else "-"
            })
        df_delta = pd.DataFrame(rows)
        st.dataframe(df_delta, use_container_width=True)

    st.write("### 🧠 Why the AI Chose This Winner")
    detailed_explanation = ""
    if probs[predicted_winner] > 80:
        detailed_explanation += f"- **{predicted_winner}** is significantly faster on the dominant lap.\n"
    if vpi[predicted_winner] > 1.05:
        detailed_explanation += f"- Reinforcement learning shows **{predicted_winner}** has strong historical performance.\n"
    if revealed_track in ["Expressway", "Highway"]:
        detailed_explanation += "- High-speed tracks strongly favor Supercar / Sports Car.\n"
    if revealed_track in ["Dirt", "Bumpy", "Potholes"]:
        detailed_explanation += "- Rough tracks often favor ORV / Monster Truck.\n"
    if detailed_explanation == "":
        detailed_explanation = "The AI combined physics, learned geometry, and historical win patterns to select this winner."
    st.info(detailed_explanation)

    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top_prob = sorted_probs[0][1]
    second_prob = sorted_probs[1][1]
    volatility = top_prob - second_prob
    if volatility < 5:
        vol_text = "High randomness — unpredictable race"
    elif volatility < 15:
        vol_text = "Moderate confidence"
    else:
        vol_text = "High confidence"
    st.metric("Prediction Confidence", f"{top_prob:.1f}%", vol_text)

    st.write("### ⏱️ Lap-by-Lap Expected Time (Physics Model)")
    lap_data = []
    for v in ctx['v']:
        for lap_idx in range(3):
            track = ctx['t'] if lap_idx == ctx['idx'] else "Hidden"
            lap_data.append({
                "Vehicle": v,
                "Lap": lap_idx + 1,
                "Track": track,
                "Speed": SPEED_DATA[v].get(track, "—") if track != "Hidden" else "—"
            })
    st.dataframe(pd.DataFrame(lap_data), use_container_width=True)

    st.write("### 🔮 Hidden Lap Predictions")
    st.caption("Based on learned Markov transitions and geometry.")
    st.json({
        "Revealed Lap": ctx['slot'],
        "Revealed Track": ctx['t'],
        "Winner": predicted_winner,
        "Probabilities": probs
    })

# ---------------------------------------------------------
# 12. ANALYTICS TABS + GHOST LAP WHAT-IF
# ---------------------------------------------------------

if not history.empty:
    st.divider()
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
        "🧪 What-If Simulator"
    ])

    # 1) PERFORMANCE DASHBOARD
    with tabs[0]:
        st.write("### 📊 Performance Insights Dashboard")
        metrics = compute_basic_metrics(history)
        if not metrics:
            st.info("Not enough data yet to compute metrics.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Global Accuracy", f"{metrics['accuracy']*100:.1f}%")
            c2.metric(
                "Mean Top Probability",
                f"{metrics['mean_top_prob']*100:.1f}%" if pd.notna(metrics['mean_top_prob']) else "N/A"
            )
            c3.metric(
                "Calibration Error |p̂ - acc|",
                f"{metrics['calib_error']*100:.2f}%" if pd.notna(metrics['calib_error']) else "N/A"
            )

            c4, c5 = st.columns(2)
            c4.metric("Brier Score (↓ better)", f"{metrics['brier']:.4f}" if pd.notna(metrics['brier']) else "N/A")
            c5.metric("Log Loss (↓ better)", f"{metrics['log_loss']:.4f}" if pd.notna(metrics['log_loss']) else "N/A")

            st.caption("Calibration Error close to 0 means probabilities match reality. Brier/Log Loss lower = sharper, better-calibrated model.")

    # 2) LEARNING CURVES
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

    # 3) CALIBRATION ANALYZER
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

    # 4) DRIFT DETECTOR
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

    # 5) VOLATILITY & FEATURE IMPORTANCE
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

    # 6) ML PATTERN BRAIN
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

    # 7) LANE TRACKER
    with tabs[6]:
        st.write("### 🚦 Win Rate by Lane Context")
        if 'Lane' in history.columns and history['Lane'].notna().any():
            lane_stats = pd.crosstab(history['Lane'], history['Actual_Winner'], normalize='index') * 100
            st.dataframe(lane_stats.style.format("{:.1f}%").background_gradient(cmap="YlOrRd", axis=1))
        else:
            st.info("Record more races to see Lane win rates.")

    # 8) DATA QUALITY CHECKER
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
                st.error("⚠️ Geometry instability detected (very high variance in lap lengths for some tracks):")
                st.dataframe(unstable, use_container_width=True)
            else:
                st.success("✅ Geometry looks reasonably stable for the current data volume.")
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

    # 9) RAW HISTORY
    with tabs[8]:
        st.write("### 📂 Race History")
        st.dataframe(history.sort_index(ascending=False), use_container_width=True)
        st.download_button(
            "⬇️ Download Race History",
            history.to_csv(index=False),
            "race_history.csv",
            mime="text/csv"
        )

    # 10) WHAT-IF ANALYSIS PANEL + GHOST LAP
    with tabs[9]:
        st.write("### 🧪 What-If Analysis Panel")
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
