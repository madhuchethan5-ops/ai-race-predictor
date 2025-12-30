import os
import numpy as np
import pandas as pd
import streamlit as st
import sqlite3
from pathlib import Path
from datetime import datetime
from streamlit_extras.grid import grid
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from collections import Counter, defaultdict

# ---------------------------------------------------------
# SQLITE DATABASE (PERSISTENT — STORED IN .streamlit/)
# ---------------------------------------------------------

DB_PATH = Path("race_history.db")
DB_PATH.parent.mkdir(exist_ok=True)

def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn
# ---------------------------------------------------------
# NORMAL INIT_DB
# ---------------------------------------------------------
def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS races (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            vehicle_1 TEXT,
            vehicle_2 TEXT,
            vehicle_3 TEXT,
            actual_winner TEXT,
            predicted_winner TEXT,
            top_prob REAL,
            was_correct REAL,
            surprise_index REAL,
            lap_1_track TEXT,
            lap_2_track TEXT,
            lap_3_track TEXT,
            lap_1_len REAL,
            lap_2_len REAL,
            lap_3_len REAL,
            lane TEXT,
            sim_predicted_winner TEXT,
            ml_predicted_winner TEXT,
            sim_top_prob REAL,
            ml_top_prob REAL,
            sim_was_correct REAL,
            ml_was_correct REAL,
            hidden_track_error_l1 REAL,
            hidden_track_error_l2 REAL,
            hidden_track_error_l3 REAL,
            hidden_len_error_l1 REAL,
            hidden_len_error_l2 REAL,
            hidden_len_error_l3 REAL,
            last_updated TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ---------------------------------------------------------
# SAVE RACE
# ---------------------------------------------------------
def save_race_to_db(row: dict):
    row_l = {k.lower(): v for k, v in row.items()}

    conn = get_connection()
    conn.execute("""
        INSERT INTO races (
            timestamp, vehicle_1, vehicle_2, vehicle_3,
            actual_winner, predicted_winner, top_prob, was_correct, surprise_index,
            lap_1_track, lap_2_track, lap_3_track,
            lap_1_len, lap_2_len, lap_3_len,
            lane,
            sim_predicted_winner, ml_predicted_winner,
            sim_top_prob, ml_top_prob,
            sim_was_correct, ml_was_correct,
            hidden_track_error_l1, hidden_track_error_l2, hidden_track_error_l3,
            hidden_len_error_l1, hidden_len_error_l2, hidden_len_error_l3,
            last_updated
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row_l.get("timestamp"),
        row_l.get("vehicle_1"), row_l.get("vehicle_2"), row_l.get("vehicle_3"),
        row_l.get("actual_winner"), row_l.get("predicted_winner"),
        row_l.get("top_prob"), row_l.get("was_correct"), row_l.get("surprise_index"),
        row_l.get("lap_1_track"), row_l.get("lap_2_track"), row_l.get("lap_3_track"),
        row_l.get("lap_1_len"), row_l.get("lap_2_len"), row_l.get("lap_3_len"),
        row_l.get("lane"),
        row_l.get("sim_predicted_winner"), row_l.get("ml_predicted_winner"),
        row_l.get("sim_top_prob"), row_l.get("ml_top_prob"),
        row_l.get("sim_was_correct"), row_l.get("ml_was_correct"),
        row_l.get("hidden_track_error_l1"), row_l.get("hidden_track_error_l2"), row_l.get("hidden_track_error_l3"),
        row_l.get("hidden_len_error_l1"), row_l.get("hidden_len_error_l2"), row_l.get("hidden_len_error_l3"),
        row_l.get("last_updated")
    ))
    conn.commit()
    conn.close()

# ---------------------------------------------------------
# LOAD HISTORY
# ---------------------------------------------------------
def load_history():
    conn = get_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM races ORDER BY id ASC", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df

history = load_history()
# Ignore imported CSV rows (they have no track data)
valid_history = history.dropna(subset=["lap_1_track", "lap_2_track", "lap_3_track"])

if len(valid_history) > 0:
    last_race = valid_history.iloc[-1]
else:
    last_race = None

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
# ---------------------------------------------------------
# SPEED DATA + CONSTANTS
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


TRACK_ALIASES = {
    "Road": "Highway",
    "road": "Highway",
    "Normal road": "Highway",
    "normal road": "Highway",
    "Normal": "Highway",
}

# ---------------------------------------------------------
# TRACK LENGTH PRIORS (mean + std from 500-race audit)
# ---------------------------------------------------------

TRACK_LENGTH_PRIORS = {
    "Expressway": {"mean": 36.4, "std": 16.8},
    "Highway":    {"mean": 32.0, "std": 15.2},
    "Dirt":       {"mean": 34.2, "std": 15.5},
    "Desert":     {"mean": 33.5, "std": 15.5},
    "Bumpy":      {"mean": 31.8, "std": 14.7},
    "Potholes":   {"mean": 32.7, "std": 13.9},
}

import numpy as np

def sample_track_length(track, rng=None, clip_min=10, clip_max=80):
    """
    Sample a realistic track length for SIM using mean + variance.
    Prevents SIM from hallucinating 0.9 confidence.
    """
    if rng is None:
        rng = np.random.default_rng()

    p = TRACK_LENGTH_PRIORS[track]
    L = rng.normal(p["mean"], p["std"])
    return float(np.clip(L, clip_min, clip_max))

# ---------------------------------------------------------
# GLOBAL VEHICLE WIN-RATE PRIORS (DEFAULTS / FALLBACK)
# ---------------------------------------------------------

# These are global win-rate priors (fractions, not %).
# You can refresh these manually from the race app when needed.
DEFAULT_VEHICLE_PRIORS = {
    "ORV":           0.418,
    "SUV":           0.399,
    "Monster Truck": 0.390,
    "Car":           0.374,
    "Stock Car":     0.368,
    "ATV":           0.311,
    "Motorcycle":    0.269,
    "Sports Car":    0.259,
    "Supercar":      0.218,
}

# ---------------------------------------------------------
# VEHICLE WIN-RATE PRIOR RESOLVER (UI + DEFAULTS, LEAK-SAFE)
# ---------------------------------------------------------

from typing import Optional, Dict, Any

def get_vehicle_win_rate(
    vehicle: str,
    user_priors: Optional[Dict[str, Dict[str, Any]]],
    default_priors: Dict[str, float],
    win_rate_min: float = 0.05,
    win_rate_max: float = 0.80,
) -> float:
    """
    Return a win_rate prior for a given vehicle.

    user_priors structure (if provided from UI):
        {
          "ORV": {"win_rate": 0.42},
          "SUV": {"win_rate": 0.38},
          ...
        }

    Logic:
    - If user provides win_rate and 0.05 <= win_rate <= 0.80 -> accept.
    - Else -> fallback to default_priors.
    - If vehicle not in default_priors -> neutral 0.33.
    """

    default_wr = default_priors.get(vehicle, 0.33)

    if user_priors is None or vehicle not in user_priors:
        return float(default_wr)

    user_wr = user_priors[vehicle].get("win_rate", None)
    if user_wr is None:
        return float(default_wr)

    try:
        user_wr = float(user_wr)
        if win_rate_min <= user_wr <= win_rate_max:
            return float(user_wr)
        else:
            return float(default_wr)
    except (ValueError, TypeError):
        return float(default_wr)

# =========================================================
# HIDDEN LAP STATS + ESTIMATOR
# =========================================================

def build_hidden_lap_stats(history: pd.DataFrame):
    """
    Build stats for hidden lap estimation.

    Assumes history columns are already normalized to snake_case:
    - actual_winner
    - lap_1_track, lap_2_track, lap_3_track
    - lap_1_len, lap_2_len, lap_3_len
    - lane
    """
    stats = {
        "global": {1: Counter(), 2: Counter(), 3: Counter()},
        "conditional": defaultdict(lambda: {1: Counter(), 2: Counter(), 3: Counter()}),
        "length": {1: [], 2: [], 3: []},
    }

    if history is None or history.empty:
        return stats

    lane_to_idx = {"Lap 1": 1, "Lap 2": 2, "Lap 3": 3}

    for _, row in history.iterrows():
        winner = row.get("actual_winner")
        if pd.isna(winner):
            continue

        lap_tracks = {
            1: row.get("lap_1_track"),
            2: row.get("lap_2_track"),
            3: row.get("lap_3_track"),
        }
        lap_lens = {
            1: row.get("lap_1_len"),
            2: row.get("lap_2_len"),
            3: row.get("lap_3_len"),
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
        revealed_idx = lane_to_idx.get(row.get("lane"))
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

        # NOTE: rest of estimate_hidden_laps (expected_len etc.) should follow below
        # in your original code; keep or paste it as-is after this point.
    
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
    Build a terrain–vehicle win-rate matrix from history.

    Returns:
        tv_matrix[(vehicle, terrain)] = win_rate (0–1)
        tv_samples[(vehicle, terrain)] = sample count
    """
    tv_counts = {}   # (vehicle, terrain) -> {"wins": x, "total": y}

    if history is None or history.empty:
        return {}, {}

    for _, row in history.iterrows():
        actual_winner = row.get("actual_winner")
        if pd.isna(actual_winner):
            continue

        vehicles = [
            row.get("vehicle_1"),
            row.get("vehicle_2"),
            row.get("vehicle_3"),
        ]

        lap_tracks = [
            row.get("lap_1_track"),
            row.get("lap_2_track"),
            row.get("lap_3_track"),
        ]

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

    tv_matrix = {}
    tv_samples = {}
    for key, stats_v in tv_counts.items():
        wins = stats_v["wins"]
        total = max(stats_v["total"], 1)
        tv_matrix[key] = wins / total
        tv_samples[key] = total

    return tv_matrix, tv_samples


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
    if total_strength > 0:
        norm_strengths = {v: strengths[v] / total_strength for v in vehicles}
    else:
        norm_strengths = {v: 1.0 / len(vehicles) for v in vehicles}

    # 3) Apply a small multiplicative adjustment to probabilities
    adjusted = {}
    for v in vehicles:
        base_p = final_probs[v]
        s = norm_strengths[v]
        factor = 1.0 + strength_alpha * (s - (1.0 / len(vehicles)))
        adjusted[v] = base_p * factor

    # 4) Renormalize to keep total probability consistent
    total_adj = sum(adjusted.values())
    if total_adj > 0:
        adjusted = {
            v: (adjusted[v] / total_adj) * sum(final_probs.values())
            for v in vehicles
        }
    else:
        adjusted = final_probs.copy()

    return adjusted, strengths

# =========================================================
# SURPRISE INDEX — How unexpected was the outcome?
# =========================================================

def compute_surprise_index(row: dict) -> float:
    """
    Returns a surprise score for a race:
    - High if winner had low predicted probability
    - Low if winner was expected

    Expects:
        row["actual_winner"]
        row["win_probs"]  (dict: vehicle -> probability in percent)
    """
    # Make it case-insensitive for robustness
    winner = row.get("actual_winner") or row.get("Actual_Winner")
    probs = row.get("win_probs") or row.get("Win_Probs")

    if not winner or not probs or winner not in probs:
        return 0.0

    p_win = probs[winner] / 100.0
    surprise = 1.0 - p_win  # 0.0 = expected, 1.0 = total shock

    return round(surprise, 3)

# ---------------------------------------------------------
# CONFIDENCE BAR
# ---------------------------------------------------------

def get_confidence_color(prob: float) -> str:
    if prob >= 70:
        return "#2e7d32"
    elif prob >= 40:
        return "#f9a825"
        # amber
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
# AUTO CLEANER
# ---------------------------------------------------------

def auto_clean_history(df: pd.DataFrame):
    """
    Clean up raw history.

    NOTE: This still expects legacy TitleCase column names and is intended
    for pre-normalization cleaning (e.g., when importing CSV). For data
    coming from SQLite, you should be using normalized snake_case already.
    """
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

def sim_meta_from_probs(sim_probs: dict):
    """
    Compute SIM meta-features from a dict {vehicle: prob_in_percent}.
    Returns normalized top, second, margin, entropy, volatility.
    """
    if not sim_probs or len(sim_probs) < 3:
        # fallback neutral
        return 0.33, 0.33, 0.0, 1.10, 0.0

    arr = np.array(list(sim_probs.values()), dtype=float)
    arr = np.clip(arr, 1e-6, None)
    arr_norm = arr / arr.sum()

    top = float(arr_norm.max())
    second = float(np.partition(arr_norm, -2)[-2])
    margin = top - second
    entropy = float(-(arr_norm * np.log(arr_norm)).sum())
    volatility = float(arr_norm.std())

    return top, second, margin, entropy, volatility

# ---------------------------------------------------------
# 4. ML FEATURE ENGINEERING (LEAK-SAFE) + TRAINING
# ---------------------------------------------------------

def estimate_ml_temperature(history_df: pd.DataFrame, calib_min_hist: int = 50) -> float:
    cols = ["ml_top_prob", "ml_was_correct"]
    if history_df is None or history_df.empty or not all(c in history_df.columns for c in cols):
        return 1.0

    recent = history_df.dropna(subset=cols).tail(200)
    if len(recent) < calib_min_hist:
        return 1.0

    avg_conf = recent["ml_top_prob"].mean()
    avg_acc = recent["ml_was_correct"].mean()
    if avg_conf <= 0 or avg_acc <= 0:
        return 1.0

    calib_error = abs(avg_conf - avg_acc)
    temp = float(np.clip(1.0 + calib_error * 2.0, 0.8, 2.0))
    return temp

def add_leakage_safe_win_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds leak-safe per-vehicle historical win rates.

    Expects df to already use snake_case columns:
    - timestamp
    - vehicle_1, vehicle_2, vehicle_3
    - actual_winner
    """
    df = df.copy()

    if "timestamp" not in df.columns:
        df["timestamp"] = pd.Timestamp.now()

    df = df.sort_values("timestamp").reset_index(drop=True)
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

        v1_rates.append(rate(row["vehicle_1"]))
        v2_rates.append(rate(row["vehicle_2"]))
        v3_rates.append(rate(row["vehicle_3"]))

        for v in [row["vehicle_1"], row["vehicle_2"], row["vehicle_3"]]:
            if pd.notna(v):
                race_counts[v] = race_counts.get(v, 0) + 1
        w = row["actual_winner"]
        if pd.notna(w):
            win_counts[w] = win_counts.get(w, 0) + 1

    # snake_case internal feature names
    df["v1_win_rate"] = v1_rates
    df["v2_win_rate"] = v2_rates
    df["v3_win_rate"] = v3_rates
    return df

def compute_live_vehicle_win_rates(history_df: pd.DataFrame, v1: str, v2: str, v3: str):
    """
    Compute per-vehicle win rates from full history, leak-safe style.
    Used for live feature building so it matches training semantics.
    """
    if history_df is None or history_df.empty:
        return 0.33, 0.33, 0.33

    df = history_df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    win_counts = {}
    race_counts = {}

    for _, row in df.iterrows():
        vs = [row.get("vehicle_1"), row.get("vehicle_2"), row.get("vehicle_3")]
        for v in vs:
            if pd.notna(v):
                race_counts[v] = race_counts.get(v, 0) + 1
        w = row.get("actual_winner")
        if pd.notna(w):
            win_counts[w] = win_counts.get(w, 0) + 1

    def rate(v):
        rc = race_counts.get(v, 0)
        wc = win_counts.get(v, 0)
        if rc > 0:
            return wc / rc
        return 0.33

    return rate(v1), rate(v2), rate(v3)

def build_pre_race_training_rows(history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform full-history rows (3 laps known) into pre-race-style rows
    using the same logic as build_single_feature_row.

    We use 'lane' to detect which lap was revealed pre-race.
    """
    if history_df is None or history_df.empty:
        return pd.DataFrame()

    rows = []
    for _, row in history_df.iterrows():
        # Basic sanity: need vehicles and actual_winner
        if any(col not in row for col in ["vehicle_1", "vehicle_2", "vehicle_3", "actual_winner"]):
            continue

        v1 = row["vehicle_1"]
        v2 = row["vehicle_2"]
        v3 = row["vehicle_3"]
        winner = row["actual_winner"]

        # Determine which lap was revealed from 'lane'
        lane = row.get("lane", None)
        if lane == "Lap 1":
            k_idx = 0
            k_type = row.get("lap_1_track", "Unknown")
        elif lane == "Lap 2":
            k_idx = 1
            k_type = row.get("lap_2_track", "Unknown")
        elif lane == "Lap 3":
            k_idx = 2
            k_type = row.get("lap_3_track", "Unknown")
        else:
            # If we don't know which lap was shown, skip this race for ML
            continue

        # Build SIM meta from stored Win_Prob_1/2/3 if available
        sim_meta_live = None
        if all(col in row for col in ["Win_Prob_1", "Win_Prob_2", "Win_Prob_3"]):
            sim_probs_hist = {
                v1: row["Win_Prob_1"],
                v2: row["Win_Prob_2"],
                v3: row["Win_Prob_3"],
            }
            sim_meta_live = sim_meta_from_probs(sim_probs_hist)

        # Build a pre-race-style feature row using the same logic as live
        feat_row = build_single_feature_row(
            v1,
            v2,
            v3,
            k_idx,
            k_type,
            history_df,
            user_vehicle_priors=None,
            sim_meta_live=sim_meta_live,
        )

        # Target: winner_idx
        vs = [v1, v2, v3]
        if winner not in vs:
            continue
        winner_idx = vs.index(winner)
        feat_row["winner_idx"] = int(winner_idx)

        rows.append(feat_row)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)
            
def build_training_data(history_df: pd.DataFrame):
    """
    Build ML training data in the same 1-known-lap regime as live prediction.
    Uses build_single_feature_row to guarantee feature alignment.
    """
    if history_df is None or history_df.empty:
        return None, None, None, None

    df = build_pre_race_training_rows(history_df)
    if df is None or df.empty:
        return None, None, None, None

    # Target
    y = df["winner_idx"].astype(int)

    # Feature columns must match build_single_feature_row output
    feature_cols = [
        "vehicle_1", "vehicle_2", "vehicle_3",
        "lap_1_track", "lap_2_track", "lap_3_track",
        "lap_1_len", "lap_2_len", "lap_3_len",
        "lane",
        "high_speed_share", "rough_share",
        "v1_win_rate", "v2_win_rate", "v3_win_rate",
        "geom_lap1_mean", "geom_lap2_mean", "geom_lap3_mean",
        "geom_lap1_std", "geom_lap2_std", "geom_lap3_std",
        "geom_range", "geom_split_flag",
        "sim_top_prob", "sim_second_prob", "sim_margin",
        "sim_entropy", "sim_volatility",
        "trans_entropy_l1", "trans_entropy_l2", "trans_entropy_l3",
    ]

    cat_features = [
        "vehicle_1", "vehicle_2", "vehicle_3",
        "lap_1_track", "lap_2_track", "lap_3_track",
        "lane",
    ]

    num_features = [
        "lap_1_len", "lap_2_len", "lap_3_len",
        "high_speed_share", "rough_share",
        "v1_win_rate", "v2_win_rate", "v3_win_rate",
        "geom_lap1_mean", "geom_lap2_mean", "geom_lap3_mean",
        "geom_lap1_std", "geom_lap2_std", "geom_lap3_std",
        "geom_range", "geom_split_flag",
        "sim_top_prob", "sim_second_prob", "sim_margin",
        "sim_entropy", "sim_volatility",
        "trans_entropy_l1", "trans_entropy_l2", "trans_entropy_l3",
    ]

    # For now, use uniform sample weights (can reintroduce surprise_weight later)
    sample_weights = np.ones(len(df), dtype=float)

    X = df[feature_cols].copy()

    return X, y, (cat_features, num_features), sample_weights

def train_ml_model(history_df: pd.DataFrame):
    df_recent = history_df.copy().tail(200)

    X, y, feat_info, sample_weights = build_training_data(df_recent)
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
        learning_rate=0.1,
    )

    model = Pipeline(steps=[
        ("pre", preprocessor),
        ("clf", clf),
    ])

    model.fit(X, y, clf__sample_weight=sample_weights)

    return model, n_samples


@st.cache_resource
def get_trained_model(history_df: pd.DataFrame):
    return train_ml_model(history_df)

# ---------------------------------------------------------
# BLEND WEIGHT HELPER (for debug panel + consistency)
# ---------------------------------------------------------
def compute_blend_weight(sim_top: float, ml_top: float) -> float:
    """
    Compute the ML blend weight using the same logic as run_full_prediction.
    """
    # Base weight from confidence gap
    gap = ml_top - sim_top

    # Start at 0.55 (balanced)
    w = 0.55 + gap * 0.35

    # Clamp to safe range
    w = float(np.clip(w, 0.40, 0.75))

    return w

# ---------------------------------------------------------
# BRIER SCORE HELPER (for debug panel + consistency)
# ---------------------------------------------------------
def compute_brier(prob_dict, actual_winner):
    """
    Compute Brier score for a 3-class probability distribution.
    prob_dict: {vehicle: prob}
    actual_winner: string
    """
    if actual_winner not in prob_dict:
        return None

    # Convert to vector
    vehicles = list(prob_dict.keys())
    probs = np.array([prob_dict[v] for v in vehicles], dtype=float)

    # One-hot target
    y = np.zeros(3)
    y[vehicles.index(actual_winner)] = 1.0

    # Brier score
    return float(np.mean((probs - y) ** 2))

# ---------------------------------------------------------
# EXPECTED REGRET HELPER
# ---------------------------------------------------------
def compute_expected_regret(sim_probs, ml_probs):
    """
    Compute expected regret between SIM and ML distributions.
    sim_probs: dict {vehicle: prob}
    ml_probs: array/list of 3 probabilities in v1,v2,v3 order
    """
    vehicles = list(sim_probs.keys())
    sim_vec = np.array([sim_probs[v] for v in vehicles], dtype=float)
    ml_vec  = np.array(ml_probs, dtype=float)

    # Normalize if needed
    sim_vec = sim_vec / sim_vec.sum()
    ml_vec  = ml_vec / ml_vec.sum()

    # Expected regret = L1 distance / 2
    regret = float(np.sum(np.abs(sim_vec - ml_vec)) / 2.0)
    return regret

# ---------------------------------------------------------
# CHAOS MODE HELPER
# ---------------------------------------------------------
def compute_chaos_mode(sim_probs, ml_probs):
    """
    Chaos mode triggers when both SIM and ML are confident
    AND they disagree on the winner.
    """
    vehicles = list(sim_probs.keys())

    sim_top = max(sim_probs.values())
    ml_top  = max(ml_probs)

    sim_winner = max(sim_probs, key=sim_probs.get)
    ml_winner  = vehicles[int(np.argmax(ml_probs))]

    chaos = (
        sim_top > 0.70 and
        ml_top  > 0.70 and
        sim_winner != ml_winner
    )
    return chaos

# ---------------------------------------------------------
# EXPECTED LENGTH ESTIMATOR (NEW, CORRECT)
# ---------------------------------------------------------

def expected_length(history_df: pd.DataFrame, lap_idx: int, track_type: str) -> float:
    """
    ML feature: expected lap length for a given terrain.
    ML uses deterministic means from TRACK_LENGTH_PRIORS.
    SIM uses variance-aware sampling.
    This keeps ML and SIM aligned without split-brain.
    """

    if track_type is None or track_type == "Unknown":
        return 33.3

    # ML uses the mean only (no history, no shrinkage)
    if track_type in TRACK_LENGTH_PRIORS:
        return TRACK_LENGTH_PRIORS[track_type]["mean"]

    return 33.3  # fallback

# ---------------------------------------------------------
# 5. SINGLE-ROW FEATURE BUILDER FOR LIVE PREDICTIONS
# ---------------------------------------------------------

def build_single_feature_row(
    v1: str,
    v2: str,
    v3: str,
    k_idx: int,
    k_type: str,
    history_df: pd.DataFrame,
    user_vehicle_priors: dict | None = None,
    sim_meta_live: tuple[float, float, float, float, float] | None = None,
) -> pd.DataFrame:
    """
    Build a single feature row for live ML prediction.

    v1, v2, v3   : vehicle names
    k_idx        : known lap index (0, 1, 2)
    k_type       : known lap track type (e.g. 'Desert', 'Expressway')
    history_df   : full normalized history, used to infer expected lap lengths
    """

    # ---------------------------------------------------------
    # KNOWN / UNKNOWN LAPS
    # ---------------------------------------------------------
    lap_tracks = ["Unknown", "Unknown", "Unknown"]
    lap_tracks[k_idx] = k_type

    lap_lens = [
        expected_length(history_df, 0, lap_tracks[0]),
        expected_length(history_df, 1, lap_tracks[1]),
        expected_length(history_df, 2, lap_tracks[2]),
    ]

    lane = f"Lap {k_idx + 1}"

    high_speed_share = (
        lap_tracks.count("Expressway") + lap_tracks.count("Highway")
    ) / 3.0

    rough_share = sum(
        1 for t in lap_tracks if t in ["Dirt", "Bumpy", "Potholes"]
    ) / 3.0

    # ---------------------------------------------------------
    # VEHICLE WIN-RATE PRIORS (LIVE, HISTORY-ALIGNED)
    # ---------------------------------------------------------
    if history_df is not None and not history_df.empty:
        v1_wr, v2_wr, v3_wr = compute_live_vehicle_win_rates(history_df, v1, v2, v3)
    else:
        v1_wr = v2_wr = v3_wr = 0.33

    # ---------------------------------------------------------
    # BASE FEATURE DICT (MUST MATCH TRAINING COLUMNS)
    # ---------------------------------------------------------
    data = {
        "vehicle_1": v1,
        "vehicle_2": v2,
        "vehicle_3": v3,

        "lap_1_track": lap_tracks[0],
        "lap_2_track": lap_tracks[1],
        "lap_3_track": lap_tracks[2],

        "lap_1_len": float(lap_lens[0]),
        "lap_2_len": float(lap_lens[1]),
        "lap_3_len": float(lap_lens[2]),

        "lane": lane,

        "high_speed_share": float(high_speed_share),
        "rough_share": float(rough_share),

        "v1_win_rate": float(v1_wr),
        "v2_win_rate": float(v2_wr),
        "v3_win_rate": float(v3_wr),
    }

    # ---------------------------------------------------------
    # GEOMETRY REGIME FEATURES (LIVE)
    # ---------------------------------------------------------
    if history_df is not None and not history_df.empty:
        geom_means = [
            history_df["lap_1_len"].mean(),
            history_df["lap_2_len"].mean(),
            history_df["lap_3_len"].mean(),
        ]
        geom_stds = [
            history_df["lap_1_len"].std(),
            history_df["lap_2_len"].std(),
            history_df["lap_3_len"].std(),
        ]
    else:
        geom_means = [33.3, 33.3, 33.3]
        geom_stds = [5.0, 5.0, 5.0]

    data["geom_lap1_mean"] = geom_means[0]
    data["geom_lap2_mean"] = geom_means[1]
    data["geom_lap3_mean"] = geom_means[2]

    data["geom_lap1_std"] = geom_stds[0]
    data["geom_lap2_std"] = geom_stds[1]
    data["geom_lap3_std"] = geom_stds[2]

    geom_range = max(geom_means) - min(geom_means)
    data["geom_range"] = geom_range
    data["geom_split_flag"] = 1 if geom_range >= 20 else 0

    # ---------------------------------------------------------
    # SIM META-FEATURES (LIVE)
    # ---------------------------------------------------------
    if sim_meta_live is not None:
        sim_top, sim_second, sim_margin, sim_entropy, sim_volatility = sim_meta_live
    else:
        # fallback neutral priors
        sim_top, sim_second, sim_margin, sim_entropy, sim_volatility = 0.33, 0.33, 0.0, 1.10, 0.0

    data["sim_top_prob"] = float(sim_top)
    data["sim_second_prob"] = float(sim_second)
    data["sim_margin"] = float(sim_margin)
    data["sim_entropy"] = float(sim_entropy)
    data["sim_volatility"] = float(sim_volatility)

    # ---------------------------------------------------------
    # TRANSITION ENTROPY (LIVE)
    # ---------------------------------------------------------
    mats = compute_transition_matrices(history_df) if history_df is not None else {}

    def entropy_from_mat(mat):
        arr = (mat.values / 100.0).astype(float)
        arr = np.clip(arr, 1e-12, None)
        return float(-(arr * np.log(arr)).sum())

    ent_l1 = ent_l2 = ent_l3 = 0.0

    if (1, 2) in mats:
        ent_l1 += entropy_from_mat(mats[(1, 2)])
    if (1, 3) in mats:
        ent_l1 += entropy_from_mat(mats[(1, 3)])

    if (2, 1) in mats:
        ent_l2 += entropy_from_mat(mats[(2, 1)])
    if (2, 3) in mats:
        ent_l2 += entropy_from_mat(mats[(2, 3)])

    if (3, 1) in mats:
        ent_l3 += entropy_from_mat(mats[(3, 1)])
    if (3, 2) in mats:
        ent_l3 += entropy_from_mat(mats[(3, 2)])

    data["trans_entropy_l1"] = ent_l1
    data["trans_entropy_l2"] = ent_l2
    data["trans_entropy_l3"] = ent_l3

    return pd.DataFrame([data])
# ---------------------------------------------------------
# 6. METRICS & MODEL SKILL
# ---------------------------------------------------------

def compute_basic_metrics(history: pd.DataFrame):
    if history.empty:
        return None

    df = history.dropna(subset=['actual_winner', 'predicted_winner'])
    if df.empty:
        return None

    acc = (df['actual_winner'] == df['predicted_winner']).mean()

    # Calibration metrics
    if 'top_prob' in df.columns and 'was_correct' in df.columns:
        cal_df = df.dropna(subset=['top_prob', 'was_correct'])
        if not cal_df.empty:
            mean_top_prob = cal_df['top_prob'].mean()
            mean_acc = cal_df['was_correct'].mean()
            calib_error = abs(mean_top_prob - mean_acc)
        else:
            mean_top_prob = np.nan
            calib_error = np.nan
    else:
        mean_top_prob = np.nan
        calib_error = np.nan

    # Brier score
    if 'top_prob' in df.columns and 'was_correct' in df.columns:
        cal_df = df.dropna(subset=['top_prob', 'was_correct'])
        if not cal_df.empty:
            brier = ((cal_df['top_prob'] - cal_df['was_correct'])**2).mean()
        else:
            brier = np.nan
    else:
        brier = np.nan

    # Log loss
    if 'top_prob' in df.columns and 'was_correct' in df.columns:
        cal_df = df.dropna(subset=['top_prob', 'was_correct'])
        if not cal_df.empty:
            eps = 1e-8
            p = np.clip(cal_df['top_prob'], eps, 1 - eps)
            y = cal_df['was_correct']
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
        'log_loss': log_loss,
    }


def compute_model_skill(history: pd.DataFrame, window: int = 100):
    cols = ['sim_top_prob', 'sim_was_correct',
            'ml_top_prob', 'ml_was_correct']
    if not all(c in history.columns for c in cols):
        return None

    df = history.dropna(subset=cols).tail(window)
    if df.empty:
        return None

    sim_brier = ((df['sim_top_prob'] - df['sim_was_correct'])**2).mean()
    ml_brier = ((df['ml_top_prob'] - df['ml_was_correct'])**2).mean()

    return {
        "sim_brier": float(sim_brier),
        "ml_brier": float(ml_brier),
        "n": int(len(df)),
    }


def compute_learning_curve(history: pd.DataFrame, window: int = 30):
    if history.empty:
        return None

    df = history.dropna(subset=['actual_winner', 'predicted_winner']).copy()
    if df.empty:
        return None

    df = df.reset_index(drop=True)
    df['Correct'] = (df['actual_winner'] == df['predicted_winner']).astype(float)

    if 'top_prob' in df.columns and 'was_correct' in df.columns:
        df2 = df.dropna(subset=['top_prob', 'was_correct']).copy()
        if df2.empty:
            df['Acc_Roll'] = df['Correct'].rolling(window).mean()
            df['Brier_Roll'] = np.nan
            return df
        df2['Brier'] = (df2['top_prob'] - df2['was_correct'])**2
        df2['Acc_Roll'] = df2['was_correct'].rolling(window).mean()
        df2['Brier_Roll'] = df2['Brier'].rolling(window).mean()
        return df2
    else:
        df['Acc_Roll'] = df['Correct'].rolling(window).mean()
        df['Brier_Roll'] = np.nan
        return df


def compute_learned_geometry(df: pd.DataFrame):
    results = []
    for lap in [1, 2, 3]:
        t_col = f"lap_{lap}_track"
        l_col = f"lap_{lap}_len"
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
            c1 = f"lap_{i}_track"
            c2 = f"lap_{j}_track"
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
            suffixes=('_early', '_late'),
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

    df = history_df.dropna(subset=['actual_winner', 'sim_was_correct'])
    if df.empty:
        return {}

    bias = df.groupby('actual_winner')['sim_was_correct'].mean().to_dict()

    corrected = {}
    for veh, score in bias.items():
        corrected[veh] = float(np.clip(1.0 + (score - 0.55) * 0.10, 0.97, 1.03))

    return corrected

def get_known_terrain_from_row(row: pd.Series) -> str:
    """
    Returns the terrain of the lap that was revealed pre-race.
    Uses the 'lane' column to determine which lap was shown.
    """
    lane = row.get("lane", None)
    if lane == "Lap 1":
        return row.get("lap_1_track", "Unknown")
    elif lane == "Lap 2":
        return row.get("lap_2_track", "Unknown")
    elif lane == "Lap 3":
        return row.get("lap_3_track", "Unknown")
    return "Unknown"

# ---------------------------------------------------------
# 7. CORE SIMULATION ENGINE (CLEAN VERSION)
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

    # 1. BAYESIAN REINFORCEMENT (VPI)
    vpi_raw = {v: 1.0 for v in vehicles}

    # history_df is normalized → use lowercase column names
    if history_df is not None and not history_df.empty and 'actual_winner' in history_df.columns:
        winners = history_df['actual_winner'].dropna()
        wins = winners.value_counts()

        veh_cols = [c for c in history_df.columns if c.startswith('vehicle_')]
        if veh_cols:
            all_veh = pd.concat([history_df[c] for c in veh_cols], axis=0).dropna()
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

    # -----------------------------------------------------
    # 2. MARKOV TRANSITIONS → lap_probs
    # -----------------------------------------------------
    lap_probs = {0: None, 1: None, 2: None}

    if history_df is not None and not history_df.empty:
        known_col = f"lap_{k_idx + 1}_track"
        if known_col in history_df.columns:
            matches = history_df[history_df[known_col] == k_type].tail(200)
            global_transitions = {}

            # GLOBAL TRANSITIONS
            for j in range(3):
                if j == k_idx:
                    continue
                from_col = f"lap_{k_idx + 1}_track"
                to_col   = f"lap_{j + 1}_track"

                if from_col in history_df.columns and to_col in history_df.columns:
                    valid = history_df[[from_col, to_col]].dropna()
                    if not valid.empty:
                        counts = valid.groupby([from_col, to_col]).size().unstack(fill_value=0)
                        if k_type in counts.index:
                            row = counts.loc[k_type]
                            arr = row.reindex(TRACK_OPTIONS, fill_value=0).astype(float)
                            arr = arr + smoothing
                            global_transitions[j] = arr / arr.sum()

            # MATCH-SPECIFIC TRANSITIONS
            for j in range(3):
                if j == k_idx:
                    continue

                t_col = f"lap_{j + 1}_track"
                if t_col in matches.columns and not matches.empty:
                    counts = matches[t_col].value_counts()
                    arr = counts.reindex(TRACK_OPTIONS, fill_value=0).astype(float)
                    arr = arr + smoothing
                    lap_probs[j] = (arr / arr.sum()).values

                # fallback to global transitions
                if lap_probs[j] is None and j in global_transitions:
                    lap_probs[j] = global_transitions[j].values

    # FINAL FALLBACK: uniform distribution for any missing lap
    for j in range(3):
        if lap_probs[j] is None:
            lap_probs[j] = np.ones(len(TRACK_OPTIONS)) / len(TRACK_OPTIONS)

    # -----------------------------------------------------
    # 3. SAMPLE TERRAIN PER LAP
    # -----------------------------------------------------
    sim_terrains = []
    for i in range(3):
        if i == k_idx:
            sim_terrains.append(np.full(iterations, k_type, dtype=object))
        else:
            p = lap_probs[i]
            if p is not None and np.isfinite(p).all() and p.sum() > 0:
                sim_terrains.append(np.random.choice(TRACK_OPTIONS, size=iterations, p=p))
            else:
                sim_terrains.append(np.random.choice(TRACK_OPTIONS, size=iterations))

    terrain_matrix = np.column_stack(sim_terrains)

    # -----------------------------------------------------
    # 4. GEOMETRY (FULL SIM — VARIANCE-AWARE LENGTH SAMPLING)
    # -----------------------------------------------------
    rng = np.random.default_rng()

    len_matrix_raw = np.zeros_like(terrain_matrix, dtype=float)

    for t in TRACK_OPTIONS:
        mask = (terrain_matrix == t)
        if mask.any():
            sampled_lengths = rng.normal(
                TRACK_LENGTH_PRIORS[t]["mean"],
                TRACK_LENGTH_PRIORS[t]["std"],
                size=mask.sum()
            )
            sampled_lengths = np.clip(sampled_lengths, 10, 80)
            len_matrix_raw[mask] = sampled_lengths

    len_sums = len_matrix_raw.sum(axis=1, keepdims=True)
    len_sums[len_sums == 0] = 1.0
    len_matrix = len_matrix_raw / len_sums

    # -----------------------------------------------------
    # 5. PHYSICS BIAS + TIME SAMPLING
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # 6. RAW WIN PROBABILITIES
    # -----------------------------------------------------
    total_times = np.vstack([results[v] for v in vehicles])
    winners = np.argmin(total_times, axis=0)

    freq = pd.Series(winners).value_counts(normalize=True).sort_index()
    raw_probs = np.array([freq.get(i, 0.0) for i in range(3)], dtype=float)
    raw_probs = np.clip(raw_probs, 1e-6, 1.0)
    raw_probs /= raw_probs.sum()

    # -----------------------------------------------------
    # 7. TEMPERATURE CALIBRATION + CLAMP (SIM-ONLY)
    # -----------------------------------------------------
    def estimate_temperature_from_history(df):
        if df is None or df.empty or 'sim_top_prob' not in df.columns or 'sim_was_correct' not in df.columns:
            return 1.2

        recent = df.dropna(subset=['sim_top_prob', 'sim_was_correct']).tail(200)
        if len(recent) < calib_min_hist:
            return 1.2

        avg_conf = float(recent['sim_top_prob'].mean())
        avg_acc  = float(recent['sim_was_correct'].mean())

        if not (0.0 < avg_conf < 1.0) or not (0.0 <= avg_acc <= 1.0):
            return 1.2

        calib_error = abs(avg_conf - avg_acc)
        base_temp   = 1.0 + 2.0 * calib_error

        # Never sharpen (<1.0). Only soften.
        return float(np.clip(base_temp, 1.0, 2.0))

    temp = estimate_temperature_from_history(history_df)

    logits = np.log(raw_probs)
    calibrated_logits = logits / temp
    calibrated_probs = np.exp(calibrated_logits)
    calibrated_probs /= calibrated_probs.sum()

    calibrated_probs = np.clip(calibrated_probs, 0.05, 0.90)
    calibrated_probs /= calibrated_probs.sum()

    sim_prob_dict = {vehicles[i]: float(calibrated_probs[i]) for i in range(3)}
    return sim_prob_dict, vpi


# ---------------------------------------------------------
# SIM WRAPPER FOR DEBUG PANEL
# ---------------------------------------------------------
def compute_sim_probs(v1: str, v2: str, v3: str, history_df: pd.DataFrame):
    """
    Wrapper to compute SIM win probabilities for the debug panel.
    Uses the existing run_simulation engine.
    """
    if history_df is None or history_df.empty:
        return {v1: 0.33, v2: 0.33, v3: 0.33}

    last = history_df.tail(1).iloc[0]
    lane = last.get("lane", "Lap 1")
    k_idx = int(lane.split(" ")[1]) - 1
    k_type = last.get(f"lap_{k_idx+1}_track", "Unknown")

    sim_probs_dict, _ = run_simulation(
        v1, v2, v3,
        k_idx,
        k_type,
        history_df
    )

    return {
        v1: float(sim_probs_dict.get(v1, 0.0)),
        v2: float(sim_probs_dict.get(v2, 0.0)),
        v3: float(sim_probs_dict.get(v3, 0.0)),
    }
# ---------------------------------------------------------
# FULL PREDICTION ENGINE (NO UI) — WITH:
# - SOFT ML CONFIDENCE CAP
# - CHAOS DISAGREEMENT MODE (SIM vs ML)
# - SOFT DOUBT RULE (uses regret_tracker)
# ---------------------------------------------------------

def soft_cap_ml_probs(ml_probs, cap_max=80.0):
    """
    Softly cap ML probabilities so that the max does not exceed cap_max (in percent),
    while preserving the relative ratios between vehicles.
    ml_probs is a dict {vehicle: prob_in_percent}.
    """
    if ml_probs is None:
        return None

    current_max = max(ml_probs.values())
    if current_max <= cap_max:
        return ml_probs

    scale = cap_max / current_max
    return {v: p * scale for v, p in ml_probs.items()}


def run_full_prediction(
    v1_sel,
    v2_sel,
    v3_sel,
    k_idx,
    k_type,
    history,
    user_vehicle_priors=None,
):

    model_skill = compute_model_skill(history)

    # ---------------------------------------------------------
    # SIMULATION PROBABILITIES
    # ---------------------------------------------------------
    sim_probs, vpi_res = run_simulation(
        v1_sel, v2_sel, v3_sel, k_idx, k_type, history
    )
    # sim_probs is in 0–1 from SIM core → convert to percent for this engine
    vehicles = [v1_sel, v2_sel, v3_sel]
    sim_probs_pct = {v: sim_probs[v] * 100.0 for v in vehicles}

    sim_meta_live = sim_meta_from_probs(sim_probs)  # meta should see 0–1, keep as-is

    # ---------------------------------------------------------
    # ML PROBABILITIES
    # ---------------------------------------------------------
    ml_probs = None
    p_ml_store = None
    ml_model, n_samples = get_trained_model(history)

    if ml_model is not None:
        X_curr = build_single_feature_row(
            v1_sel,
            v2_sel,
            v3_sel,
            k_idx,
            k_type,
            history,
            user_vehicle_priors=user_vehicle_priors,
            sim_meta_live=sim_meta_live,
        )
        raw_proba = ml_model.predict_proba(X_curr)[0]

        # Temperature scaling
        ml_temp = estimate_ml_temperature(history)
        logits = np.log(np.clip(raw_proba, 1e-12, 1.0))
        scaled_logits = logits / ml_temp
        calib_proba = np.exp(scaled_logits)
        calib_proba /= calib_proba.sum()

        ml_probs = {
            v1_sel: float(calib_proba[0] * 100.0),
            v2_sel: float(calib_proba[1] * 100.0),
            v3_sel: float(calib_proba[2] * 100.0),
        }

        # Soft cap ML confidence
        ml_probs = soft_cap_ml_probs(ml_probs, cap_max=80.0)
        p_ml_store = ml_probs

    # ---------------------------------------------------------
    # BASE BLEND (ML dominant)
    # ---------------------------------------------------------
    blend_weight = 0.65  # 65% ML, 35% SIM

    if ml_probs is not None and model_skill is not None:
        sim_brier = model_skill["sim_brier"]
        ml_brier = model_skill["ml_brier"]
        n_skill = model_skill["n"]

        if n_skill >= 25 and np.isfinite(sim_brier) and np.isfinite(ml_brier):
            improvement = (sim_brier - ml_brier) / max(sim_brier, 1e-8)
            improvement_clipped = float(np.clip(improvement, -0.20, 0.20))
            blend_weight = 0.60 + 0.15 * improvement_clipped

    blend_weight = float(np.clip(blend_weight, 0.40, 0.75))

    # ---------------------------------------------------------
    # INITIAL BLENDED PROBS
    # ---------------------------------------------------------
    if ml_probs is not None and sim_probs is not None:
        blended_probs = {
            v: blend_weight * ml_probs[v] + (1.0 - blend_weight) * sim_probs_pct[v]
            for v in [v1_sel, v2_sel, v3_sel]
        }
    elif ml_probs is not None:
        blended_probs = ml_probs
    elif sim_probs is not None:
        blended_probs = sim_probs_pct
    else:
        blended_probs = {
            v1_sel: 33.33,
            v2_sel: 33.33,
            v3_sel: 33.33,
        }

    # ---------------------------------------------------------
    # SOFT DOUBT RULE (AFTER BASE BLEND)
    # ---------------------------------------------------------
    if ml_probs is not None and sim_probs_pct is not None:
        candidate_winner = max(blended_probs, key=blended_probs.get)
        dominant_terrain = k_type
        bucket_key = f"{candidate_winner}|{dominant_terrain}"

        regret_tracker = st.session_state.get("regret_tracker", {})
        regret_count = regret_tracker.get(bucket_key, 0)

        if regret_count >= 3:
            blend_weight = max(blend_weight - 0.05, 0.40)
            blended_probs = {
                v: blend_weight * ml_probs[v] + (1.0 - blend_weight) * sim_probs_pct[v]
                for v in [v1_sel, v2_sel, v3_sel]
            }

    final_probs = blended_probs

    # ---------------------------------------------------------
    # CHAOS DISAGREEMENT MODE
    # ---------------------------------------------------------
    sim_top_prob = None
    sim_winner = None
    if sim_probs_pct is not None:
        sim_winner = max(sim_probs_pct, key=sim_probs_pct.get)
        sim_top_prob = sim_probs_pct[sim_winner]  # percent

    ml_top_prob = None
    ml_winner = None
    if ml_probs is not None:
        ml_winner = max(ml_probs, key=ml_probs.get)
        ml_top_prob = ml_probs[ml_winner]

    if (
        sim_top_prob is not None and ml_top_prob is not None
        and sim_top_prob > 70.0 and ml_top_prob > 70.0
        and sim_winner != ml_winner
    ):
        ordered = sorted(final_probs.items(), key=lambda x: x[1], reverse=True)
        (v_top, p_top), (v_mid, p_mid), (v_low, p_low) = ordered

        gap = p_top - p_mid
        reduced_gap = 0.60 * gap

        new_p_top = p_mid + reduced_gap
        new_p_mid = p_mid
        new_p_low = p_low

        total = new_p_top + new_p_mid + new_p_low
        scale = 100.0 / total if total > 0 else 1.0

        final_probs = {
            v_top: new_p_top * scale,
            v_mid: new_p_mid * scale,
            v_low: new_p_low * scale,
        }

    # ---------------------------------------------------------
    # TERRAIN–VEHICLE ADJUSTMENT
    # ---------------------------------------------------------
    tv_matrix, tv_samples = build_tv_matrix(history)
    tracks = [k_type, k_type, k_type]

    ctx = {
        "v": [v1_sel, v2_sel, v3_sel],
        "idx": k_idx,
        "t": k_type,
        "slot": f"Lap {k_idx + 1}",
        "tracks": tracks,
    }

    final_probs, tv_strengths = apply_tv_adjustment(
        final_probs, ctx, tv_matrix, k_type, strength_alpha=0.15
    )

    # ---------------------------------------------------------
    # WINNER & META
    # ---------------------------------------------------------
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
    # HIDDEN LAP ESTIMATION
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
        TRACK_OPTIONS,
    )

    # ---------------------------------------------------------
    # RETURN RESULT
    # ---------------------------------------------------------
    res = {
        "p": final_probs,
        "vpi": vpi_res,
        "ctx": {
            "v": [v1_sel, v2_sel, v3_sel],
            "idx": k_idx,
            "t": k_type,
            "slot": f"Lap {k_idx + 1}",
            "tracks": tracks,
        },
        "p_sim": sim_probs_pct,
        "p_ml": p_ml_store,
        "meta": {
            "top_vehicle": top_vehicle,
            "top_prob": p1,
            "second_prob": p2,
            "volatility_gap_pp": vol_gap,
            "volatility_label": vol_label,
            "bet_safety": bet_safety,
            "expected_regret": expected_regret,
            "blend_weight_ml": blend_weight if ml_probs is not None else 0.0,
            "sim_top_prob": sim_top_prob,
            "ml_top_prob": ml_top_prob,
            "sim_winner": sim_winner,
            "ml_winner": ml_winner,
        },
        "hidden_guess": lap_guess,
        "tv_strengths": tv_strengths,
        "tv_matrix": tv_matrix,
        "tv_samples": tv_samples,
    }
    return res
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

if "regret_tracker" not in st.session_state:
    st.session_state.regret_tracker = {}
# ---------------------------------------------------------
# ALWAYS INITIALIZE PRIORS BEFORE ANY UI BLOCKS
# ---------------------------------------------------------
ui_vehicle_priors = st.session_state.get("ui_vehicle_priors", None)

# ---------------------------------------------------------
# QUADRANT LAYOUT (2×2, AUTO-FIT)
# ---------------------------------------------------------

top_left, top_right = st.columns(2)
bottom_left, bottom_right = st.columns(2)

Q1 = top_left.container()
Q2 = top_right.container()
Q3 = bottom_left.container()
Q4 = bottom_right.container()

# ---------------------------------------------------------
# ADVANCED: LIVE VEHICLE WIN-RATE INPUTS (TOP-LEVEL)
# ---------------------------------------------------------

with st.expander("Advanced: Update live vehicle win-rates (optional)"):

    st.markdown("Enter live win-rates (%) from the race app. Example: `42.3`")

    temp_wr_inputs = {}
    cols = st.columns(3)
    vehicles = list(DEFAULT_VEHICLE_PRIORS.keys())

    for i, veh in enumerate(vehicles):
        col = cols[i % 3]
        default_percent = DEFAULT_VEHICLE_PRIORS[veh] * 100.0

        val = col.text_input(
            label=veh,
            value=f"{default_percent:.1f}",
            key=f"wr_input_{veh.replace(' ', '_')}",
        )

        temp_wr_inputs[veh] = val

    if st.button("Submit Win-Rate Updates"):
        ui_vehicle_priors = {}

        for veh, val in temp_wr_inputs.items():
            try:
                wr_fraction = float(val) / 100.0
                ui_vehicle_priors[veh] = {"win_rate": wr_fraction}
            except:
                ui_vehicle_priors[veh] = {"win_rate": None}

        st.session_state["ui_vehicle_priors"] = ui_vehicle_priors
        st.success("Win-rate priors updated!")

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
            label="Select Lap",                      # <-- FIXED
            options=["Lap 1", "Lap 2", "Lap 3"],
            index=["Lap 1", "Lap 2", "Lap 3"].index(st.session_state.selected_lap)
            if st.session_state.selected_lap else 0,
            horizontal=True,
            label_visibility="collapsed"             # <-- FIXED
        )

    with terrain_col:
        st.caption("Terrain")
        terrain = st.selectbox(
            label="Select Terrain",                  # <-- FIXED
            options=list(TERRAIN_ICONS.keys()),
            index=list(TERRAIN_ICONS.keys()).index(st.session_state.selected_terrain)
            if st.session_state.selected_terrain else 0,
            label_visibility="collapsed"             # <-- FIXED
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
# Q2 — COMPACT PREDICTION PANEL (2×2 DASHBOARD LAYOUT)
# (Optimised: prediction once, diagnostics manual + cached)
# ---------------------------------------------------------

# 1) Cached helper for diagnostics (place this at top level, not inside Q2)
@st.cache_data
def compute_q2_diagnostics(res: dict):
    """
    Heavy-ish Q2 diagnostics based on a single prediction result.
    Cached so re-opening the expander is instant.
    """
    diag = {}

    # Basic check
    if res is None:
        return diag

    probs = res["p"]
    predicted_winner = max(probs, key=probs.get)

    # SIM/ML winners (if available)
    p_sim = res.get("p_sim")
    p_ml = res.get("p_ml")
    if p_sim and p_ml:
        sim_winner = max(p_sim, key=p_sim.get)
        ml_winner = max(p_ml, key=p_ml.get)
        diag["agreement"] = {
            "sim_winner": sim_winner,
            "ml_winner": ml_winner,
            "divergent": sim_winner != ml_winner,
        }
    else:
        diag["agreement"] = None

    # Context snapshot
    diag["context_snapshot"] = {
        "Revealed Lap": res["ctx"]["slot"],
        "Revealed Track": res["ctx"]["t"],
        "Winner": predicted_winner,
        "Probabilities": probs,
    }

    return diag


# ---------------------------------------------------------
# Q2 PANEL
# ---------------------------------------------------------
with Q2:
    st.markdown("### 📡 Prediction & Bet Guidance")

    # -----------------------------------------------------
    # PREDICTION TRIGGER (run_full_prediction ONCE)
    # -----------------------------------------------------
    if st.session_state.get("trigger_prediction", False):

        # Clear stale Save-form widget state BEFORE prediction
        for k in [
            "lap1_track", "lap2_track", "lap3_track",
            "lap1_len", "lap2_len", "lap3_len",
            "actual_winner"
        ]:
            if k in st.session_state:
                del st.session_state[k]

        # Build prediction context
        lap_map = {"Lap 1": 0, "Lap 2": 1, "Lap 3": 2}
        k_idx = lap_map[st.session_state.selected_lap]
        k_type = st.session_state.selected_terrain
        v1, v2, v3 = st.session_state.selected_vehicles

        # Run prediction once and store result
        st.session_state.res = run_full_prediction(
            v1,
            v2,
            v3,
            k_idx,
            k_type,
            history,
            user_vehicle_priors=ui_vehicle_priors,
        )

        # Reset trigger so we don't recompute on every rerun
        st.session_state.trigger_prediction = False

    # -----------------------------------------------------
    # DISPLAY PANEL
    # -----------------------------------------------------
    if "res" not in st.session_state:
        st.info("Set up the race on the left and run a prediction.")
    else:
        res = st.session_state["res"]
        meta = res["meta"]
        probs = res["p"]
        vpi = res["vpi"]

        # 2×2 grid layout
        col_left, col_right = st.columns(2)

        # -----------------------------------------------------
        # TOP‑LEFT: Accuracy + Winner
        # -----------------------------------------------------
        with col_left:
            st.markdown("#### 🎯 Accuracy & Winner")

            # Accuracy: cheap, but still guarded
            if not history.empty and "actual_winner" in history.columns:
                valid = history.dropna(subset=["actual_winner", "predicted_winner"])
                if not valid.empty:
                    acc = (
                        valid["predicted_winner"] == valid["actual_winner"]
                    ).mean() * 100
                    st.metric("AI Accuracy", f"{acc:.1f}%")

            predicted_winner = max(probs, key=probs.get)
            st.metric("🏆 Predicted Winner", predicted_winner)

        # -----------------------------------------------------
        # TOP‑RIGHT: Win Probabilities
        # -----------------------------------------------------
        with col_right:
            st.markdown("#### 📊 Win Probabilities")

            p_sim = res.get("p_sim")
            p_ml = res.get("p_ml")
            blend_w = res.get("meta", {}).get("blend_weight_ml")

            for v in res["ctx"]["v"]:
                p_final = probs[v]
                line = f"**{v}**: {p_final:.1f}%"

                if p_sim and p_ml:
                    line += f" (SIM {p_sim[v]:.1f}%, ML {p_ml[v]:.1f}%)"

                st.markdown(f"- {line}")
                confidence_bar(v, p_final)

            if p_sim and p_ml and blend_w is not None:
                st.caption(
                    f"Blend weight → ML: {blend_w:.2f}, SIM: {1 - blend_w:.2f}"
                )

        # -----------------------------------------------------
        # MID‑LEFT: Volatility & Safety
        # -----------------------------------------------------
        with col_left:
            st.markdown("#### ⚡ Volatility & Safety")
            st.write(f"Volatility Gap: **{meta['volatility_gap_pp']} pp**")
            st.write(f"Market: **{meta['volatility_label']}**")

            safety = meta["bet_safety"]
            if safety == "AVOID":
                st.error("**AVOID** — Too volatile or low-confidence.")
            elif safety == "CAUTION":
                st.warning("**CAUTION** — Edge exists but uncertainty is high.")
            else:
                st.success("**FAVORABLE** — Strong, stable edge detected.")

        # -----------------------------------------------------
        # MID‑RIGHT: Terrain–vehicle matchup (with dropdown)
        # -----------------------------------------------------
        with col_right:
            st.markdown("#### 🧬 Terrain–vehicle matchup")

            tv_matrix = res.get("tv_matrix", {})
            tv_samples = res.get("tv_samples", {})
            terrain_options = ["Desert", "Expressway", "Bumpy", "Dirt", "Highway", "Potholes"]
            selected_terrain = st.selectbox("Inspect tendencies for:", terrain_options)

            if tv_matrix:
                for v in res["ctx"]["v"]:
                    key = (v, selected_terrain)
                    win_rate = tv_matrix.get(key, 0.5)
                    count = tv_samples.get(key, 0)

                    if count < 10:
                        flavor = "insufficient data"
                        icon = "⚪"
                    elif win_rate >= 0.60:
                        flavor = "favorable"
                        icon = "🟢"
                    elif win_rate <= 0.40:
                        flavor = "unfavorable"
                        icon = "🔴"
                    else:
                        flavor = "neutral"
                        icon = "🟡"

                    st.markdown(
                        f"- {icon} **{v}** on **{selected_terrain}** → "
                        f"{flavor} (win rate {win_rate*100:.1f}%, n={count})"
                    )
            else:
                st.caption("Not enough history yet to learn terrain–vehicle strengths.")

        # -----------------------------------------------------
        # BOTTOM‑LEFT: Hidden Lap Guess (display only, no heavy compute)
        # -----------------------------------------------------
        with col_left:
            lg = res.get("hidden_guess")
            if lg:
                with st.expander("🤫 AI guess for hidden laps"):
                    TERRAIN_EMOJI = {
                        "Desert": "🏜️",
                        "Bumpy": "🪨",
                        "Expressway": "🛣️",
                        "Highway": "🚗",
                        "Dirt": "🌾",
                        "Potholes": "🕳️",
                    }

                    summary_lines = []

                    for k in (1, 2, 3):
                        label = f"Lap {k}"

                        # Revealed lap
                        if k == res["ctx"]["idx"] + 1:
                            st.markdown(f"**{label} (revealed):** {res['ctx']['t']}")
                            continue

                        # Hidden lap info
                        info = lg[k]
                        probs_k = info["track_probs"]
                        expected_len = info["expected_len"]

                        # Sort terrains by probability
                        sorted_probs = sorted(
                            probs_k.items(), key=lambda x: x[1], reverse=True
                        )
                        top_terrain, top_prob = sorted_probs[0]

                        emoji = TERRAIN_EMOJI.get(top_terrain, "🌍")

                        summary_lines.append(
                            f"**Lap {k}** → {emoji} **{top_terrain}‑heavy** (~{top_prob*100:.0f}%)"
                        )

                        # Top 3 terrains
                        top_str = ", ".join(
                            [
                                f"{TERRAIN_EMOJI.get(t, '🌍')} {t}: {p*100:.1f}%"
                                for t, p in sorted_probs[:3]
                            ]
                        )

                        # Safe handling of any type
                        try:
                            expected_val = float(expected_len)
                            expected_text = f"{expected_val:.1f}%"
                        except Exception:
                            expected_text = "unknown"

                        st.markdown(
                            f"**{label} (hidden):** expected length ≈ {expected_text}, "
                            f"top terrains → {top_str}"
                        )

                    # Summary section
                    st.markdown("### 🧭 Summary")
                    for line in summary_lines:
                        st.markdown(f"- {line}")
            else:
                st.write("Not enough history to estimate hidden laps.")

        # -----------------------------------------------------
        # BOTTOM‑RIGHT: Tightness + Regret
        # -----------------------------------------------------
        with col_right:
            st.markdown("#### 📈 Race Metrics")

            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            (_, p1), (_, p2) = sorted_probs[0], sorted_probs[1]
            margin = p1 - p2
            tightness = max(0, 100 - margin)

            c1, c2, c3 = st.columns(3)
            c1.metric("Race Tightness", f"{tightness:.1f}")
            c2.metric("Top‑2 Margin", f"{margin:.1f} pts")
            c3.metric("Expected Regret", f"{meta['expected_regret']:.2f}")

        # -----------------------------------------------------
        # DIAGNOSTICS (manual + cached)
        # -----------------------------------------------------
        st.markdown("---")
        with st.expander("🔍 Detailed diagnostics"):
            # Manual trigger to avoid even cached lookup unless you care
            if st.button("Compute Q2 diagnostics", key="btn_q2_diag"):
                diag = compute_q2_diagnostics(res)

                agreement = diag.get("agreement")
                if agreement:
                    if agreement["divergent"]:
                        st.warning(
                            f"⚠️ **Model Divergence:** Physics → {agreement['sim_winner']}, "
                            f"ML → {agreement['ml_winner']}. This race has higher uncertainty."
                        )
                    else:
                        st.success("✅ Physics and ML agree on the winner.")
                else:
                    st.info("SIM/ML probability breakdown not available for this run.")

                st.markdown("**Context snapshot:**")
                st.json(diag["context_snapshot"])
            else:
                st.caption("Click the button above to compute diagnostics for this race.")

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

        revealed_lap = ctx['idx']
        revealed_track = ctx['t']
        revealed_slot = ctx['slot']

        # 🔹 Use prediction context: revealed lap gets the true terrain, others default
        predicted_tracks = ["Bumpy", "Bumpy", "Bumpy"]
        predicted_tracks[revealed_lap] = revealed_track

        # 🔹 Force widget state for the revealed lap
        if revealed_lap == 0:
            st.session_state[f"lap1_track_{revealed_lap}_{revealed_track}"] = predicted_tracks[0]
        elif revealed_lap == 1:
            st.session_state[f"lap2_track_{revealed_lap}_{revealed_track}"] = predicted_tracks[1]
        elif revealed_lap == 2:
            st.session_state[f"lap3_track_{revealed_lap}_{revealed_track}"] = predicted_tracks[2]

        st.caption(
            f"Last prediction: **{predicted_winner}** on {revealed_slot} ({revealed_track})"
        )    
    else:
        st.info("Run a prediction first to enable saving.")
        # 🔹 When no prediction yet, just use neutral defaults
        predicted_tracks = ["Bumpy", "Bumpy", "Bumpy"]
    
    # Safe index helper (robust to spacing / case)
    def safe_index(value, options):
        if value is None:
            return 0
        val_norm = str(value).strip().lower()
        for i, opt in enumerate(options):
            opt_norm = str(opt).strip().lower()
            if opt_norm == val_norm:
                return i
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

            # Lock only the revealed lap
            disabled_1 = disabled_form or (prediction_available and revealed_lap == 0)
            disabled_2 = disabled_form or (prediction_available and revealed_lap == 1)
            disabled_3 = disabled_form or (prediction_available and revealed_lap == 2)

            # Use dynamic keys so Streamlit actually applies new defaults when prediction changes
            lap1_key = f"lap1_track_{revealed_lap}_{revealed_track}" if prediction_available else "lap1_track"
            lap2_key = f"lap2_track_{revealed_lap}_{revealed_track}" if prediction_available else "lap2_track"
            lap3_key = f"lap3_track_{revealed_lap}_{revealed_track}" if prediction_available else "lap3_track"

            # -----------------------------
            # LAP 1
            # -----------------------------
            with c1:
                s1t = st.selectbox(
                    "Lap 1 Track",
                    TRACK_OPTIONS,
                    index=safe_index(predicted_tracks[0], TRACK_OPTIONS),
                    disabled=disabled_1,
                    key=lap1_key,
                )
                s1l = st.number_input("Lap 1 %", 1, 100, 33, disabled=disabled_form)

            # -----------------------------
            # LAP 2
            # -----------------------------
            with c2:
                s2t = st.selectbox(
                    "Lap 2 Track",
                    TRACK_OPTIONS,
                    index=safe_index(predicted_tracks[1], TRACK_OPTIONS),
                    disabled=disabled_2,
                    key=lap2_key,
                )
                s2l = st.number_input("Lap 2 %", 1, 100, 33, disabled=disabled_form)

            # -----------------------------
            # LAP 3
            # -----------------------------
            with c3:
                s3t = st.selectbox(
                    "Lap 3 Track",
                    TRACK_OPTIONS,
                    index=safe_index(predicted_tracks[2], TRACK_OPTIONS),
                    disabled=disabled_3,
                    key=lap3_key,
                )
                s3l = st.number_input("Lap 3 %", 1, 100, 34, disabled=disabled_form)

            # Submit button
            save_clicked = st.form_submit_button("💾 Save & Train")

        # -----------------------------
        # SAVE LOGIC (SQLITE VERSION)
        # -----------------------------
        if save_clicked:

            if not prediction_available:
                st.error("Run a prediction first.")
                st.stop()

            if winner is None:
                st.error("Please select the actual winner.")
                st.stop()

            s1l, s2l, s3l = float(s1l), float(s2l), float(s3l)
            if abs((s1l + s2l + s3l) - 100) > 0.001:
                st.error("Lap lengths must total 100%.")
                st.stop()

            if not s1t or not s2t or not s3t:
                st.error("All laps must have a track selected.")
                st.stop()

            st.session_state['last_train_probs'] = dict(predicted)

            # ---------------------------------------------------------
            # Hidden-lap guess error (AI learning from mistakes)
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

                    try:
                        exp_len = float(lg[k]["expected_len"])
                    except Exception:
                        exp_len = None

                    try:
                        act_len = float(actual_lens[k])
                    except Exception:
                        act_len = None

                    if exp_len is None or act_len is None:
                        len_err[k] = None
                    else:
                        len_err[k] = abs(exp_len - act_len)

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
                # Vehicles
                'vehicle_1': ctx['v'][0],
                'vehicle_2': ctx['v'][1],
                'vehicle_3': ctx['v'][2],

                # Laps
                'lap_1_track': s1t,
                'lap_1_len': s1l,
                'lap_2_track': s2t,
                'lap_2_len': s2l,
                'lap_3_track': s3t,
                'lap_3_len': s3l,

                # Core outcome
                'predicted_winner': predicted_winner,
                'actual_winner': winner,
                'lane': revealed_slot,

                # Overall prob & correctness
                'top_prob': p1,
                'was_correct': was_correct,
                'surprise_index': surprise,

                # Physics vs ML diagnostics
                'sim_predicted_winner': sim_pred_winner,
                'ml_predicted_winner': ml_pred_winner,
                'sim_top_prob': sim_top_prob,
                'ml_top_prob': ml_top_prob,
                'sim_was_correct': sim_correct,
                'ml_was_correct': ml_correct,

                # Hidden-lap errors
                'hidden_track_error_l1': track_err[1],
                'hidden_track_error_l2': track_err[2],
                'hidden_track_error_l3': track_err[3],
                'hidden_len_error_l1': len_err[1],
                'hidden_len_error_l2': len_err[2],
                'hidden_len_error_l3': len_err[3],

                # Timestamps
                'timestamp': datetime.now().isoformat(timespec="seconds"),
                'last_updated': datetime.utcnow().isoformat(timespec="seconds"),
            }

            # ---------------------------------------------------------
            # REGRET TRACKER UPDATE (POST-MORTEM, AFTER WE KNOW THE RESULT)
            # Regime key: winner | known_terrain (revealed lap terrain)
            # ---------------------------------------------------------
            try:
                pred_winner = row['predicted_winner']
                known_terrain = get_known_terrain_from_row(row)
            
                bucket_key = f"{pred_winner}|{known_terrain}"
            
                regret_case = (
                    row['top_prob'] >= 0.60
                    and row['was_correct'] == 0.0
                    and row['surprise_index'] >= 0.40
                )
            
                if regret_case:
                    if "regret_tracker" not in st.session_state:
                        st.session_state.regret_tracker = {}
                    st.session_state.regret_tracker[bucket_key] = (
                        st.session_state.regret_tracker.get(bucket_key, 0) + 1
                    )
            
            except Exception as e:
                st.warning(f"Regret tracker update failed: {e}")
            
            save_race_to_db(row)

            st.success("✅ Race saved to database! Model will update on next prediction.")
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
        if 'was_correct' in history.columns and 'top_prob' in history.columns:
            df = history.dropna(subset=['was_correct', 'top_prob']).copy()
            if len(df) >= 20:
                window = min(20, len(df))
                df["Rolling_Accuracy"] = df["was_correct"].rolling(window).mean()
                df["Brier"] = (df["top_prob"] - df["was_correct"])**2
                df["Rolling_Brier"] = df["Brier"].rolling(window).mean()

                acc_now = df["Rolling_Accuracy"].iloc[-1]
                brier_now = df["Rolling_Brier"].iloc[-1]

                c1, c2 = st.columns(2)
                c1.metric("Recent Accuracy (20)", f"{acc_now*100:.1f}%")
                c2.metric("Recent Brier (20)", f"{brier_now:.3f}")

        st.caption("For full chaos, drift, and heatmap views, use the Analytics tabs below.")

        #st.subheader("Regret Tracker")
        #st.write("Regret tracker:", st.session_state.regret_tracker)

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
    
        if 'top_prob' in history.columns and 'was_correct' in history.columns:
            cal_df = history.dropna(subset=['top_prob', 'was_correct']).copy()
    
            if cal_df.empty:
                st.info("Not enough calibrated predictions yet.")
            else:
                cal_df['Bucket'] = (cal_df['top_prob'] * 10).astype(int) / 10.0
    
                calib_table = cal_df.groupby('Bucket').agg(
                    mean_prob=('top_prob', 'mean'),
                    emp_acc=('was_correct', 'mean'),
                    count=('was_correct', 'size')
                ).reset_index()
    
                st.write("#### Reliability Table")
                st.dataframe(
                    calib_table.style.format({'mean_prob': '{:.2f}', 'emp_acc': '{:.2f}'})
                )
    
                st.line_chart(
                    calib_table.set_index('Bucket')[['mean_prob', 'emp_acc']],
                    height=300
                )
                st.caption("If the lines track each other closely, the AI is well-calibrated.")
        else:
            st.info("top_prob / was_correct not available yet for calibration analysis.")
    
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
    
        if 'lap_1_track' in history.columns and 'lap_2_track' in history.columns:
            m = pd.crosstab(history['lap_1_track'], history['lap_2_track'], normalize='index') * 100
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
    
        if 'lane' in history.columns and history['lane'].notna().any():
            lane_stats = pd.crosstab(history['lane'], history['actual_winner'], normalize='index') * 100
            st.dataframe(lane_stats.style.format("{:.1f}%").background_gradient(cmap="YlOrRd", axis=1))
        else:
            st.info("Record more races to see Lane win rates.")
    
    # ---------------------------------------------------------
    # CSV HEALTH CHECK FUNCTION (UPDATED TO SNAKE_CASE)
    # ---------------------------------------------------------
    def csv_health_check(df: pd.DataFrame):
        issues = []
        if df is None or df.empty:
            issues.append("CSV is empty or failed to load.")
            return issues
    
        required_cols = [
            "vehicle_1", "vehicle_2", "vehicle_3",
            "lap_1_track", "lap_2_track", "lap_3_track",
            "lap_1_len", "lap_2_len", "lap_3_len",
            "actual_winner", "predicted_winner"
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

# ---------------------------------------------------------
# 🛠️ ADMIN UTILITIES (Moved to end of page)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🛠️ Admin Utilities")

# ---------------------------------------------------------
# RESET DB SCHEMA
# ---------------------------------------------------------
with st.expander("⚙️ Reset Database Schema"):
    if st.button("🧨 Drop and recreate 'races' table"):
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("DROP TABLE IF EXISTS races")
            conn.commit()
            conn.close()
            st.success("Dropped table. Reload the app to recreate with correct schema.")
        except Exception as e:
            st.error(f"Error: {e}")

# ---------------------------------------------------------
# DELETE DB FILE
# ---------------------------------------------------------
with st.expander("🗑️ Delete SQLite DB File"):
    if st.button("🧨 Force delete race_history.db"):
        try:
            os.remove(DB_PATH)
            st.success("Deleted persistent DB file.")
        except FileNotFoundError:
            st.warning("DB file not found.")
        except Exception as e:
            st.error(f"Error deleting DB: {e}")

# ---------------------------------------------------------
# IMPORT LEGACY CSV
# ---------------------------------------------------------
with st.expander("📥 Import Legacy Race History"):
    uploaded_csv = st.file_uploader("Upload old_history.csv", type=["csv"])

    if uploaded_csv is not None:
        if st.button("📥 Import CSV into Database"):
            try:
                df = pd.read_csv(uploaded_csv)

                # Normalize column names
                df.columns = [str(c).strip().lower() for c in df.columns]

                # 🔥 FIX 1: Drop unwanted index/id/unnamed columns
                df = df.loc[:, ~df.columns.str.contains('^unnamed', case=False)]
                df = df.loc[:, df.columns != ""]
                df = df.loc[:, df.columns != "id"]

                # Required columns for DB schema
                required = [
                    "timestamp", "vehicle_1", "vehicle_2", "vehicle_3",
                    "actual_winner", "predicted_winner",
                    "top_prob", "was_correct", "surprise_index",
                    "lap_1_track", "lap_2_track", "lap_3_track",
                    "lap_1_len", "lap_2_len", "lap_3_len",
                    "lane",
                    "sim_predicted_winner", "ml_predicted_winner",
                    "sim_top_prob", "ml_top_prob",
                    "sim_was_correct", "ml_was_correct",
                    "hidden_track_error_l1", "hidden_track_error_l2", "hidden_track_error_l3",
                    "hidden_len_error_l1", "hidden_len_error_l2", "hidden_len_error_l3",
                    "last_updated"
                ]

                # 🔥 FIX 2: Add missing columns with None
                for col in required:
                    if col not in df.columns:
                        df[col] = None

                # 🔥 FIX 3: Keep only columns that exist in DB schema
                df = df[required]

                # Insert into SQLite
                conn = sqlite3.connect(DB_PATH)
                df.to_sql("races", conn, if_exists="append", index=False)
                conn.close()

                st.success("CSV imported into persistent SQLite DB.")

            except Exception as e:
                st.error(f"Import failed: {e}")

# ---------------------------------------------------------
# ONE-TIME FIX: REGENERATE PHYSICS HISTORY (DISABLED)
# ---------------------------------------------------------
# def regenerate_sim_history(df):
#     """
#     Re-runs the NEW Simulation engine on all historical races.
#     Populates 'Win_Prob_1', 'Win_Prob_2', 'Win_Prob_3' so ML can see Physics.
#     """
#     if df is None or df.empty:
#         return df
#
#     st.info(f"⚡ Regenerating Physics for {len(df)} races... This creates the 'Track Affinity' features for ML.")
#     
#     prog_bar = st.progress(0)
#     new_rows = []
#     df_out = df.copy()
#
#     for idx, row in df_out.iterrows():
#         prog_bar.progress((idx + 1) / len(df_out))
#         
#         v1, v2, v3 = row['vehicle_1'], row['vehicle_2'], row['vehicle_3']
#         k_type = row['lap_1_track'] 
#         
#         try:
#             sim_res, _ = run_simulation(
#                 v1, v2, v3, 
#                 0, k_type, 
#                 df 
#             )
#             
#             df_out.at[idx, 'Win_Prob_1'] = sim_res.get(v1, 33.3)
#             df_out.at[idx, 'Win_Prob_2'] = sim_res.get(v2, 33.3)
#             df_out.at[idx, 'Win_Prob_3'] = sim_res.get(v3, 33.3)
#             
#         except Exception as e:
#             df_out.at[idx, 'Win_Prob_1'] = 33.3
#             df_out.at[idx, 'Win_Prob_2'] = 33.3
#             df_out.at[idx, 'Win_Prob_3'] = 33.3
#             
#     st.success("✅ Physics History Regenerated! ML model is now 'Track Aware'.")
#     return df_out

# ---------------------------------------------------------
# UI BUTTON TO TRIGGER IT (DISABLED)
# ---------------------------------------------------------
# if st.sidebar.button("🔧 Repair: Regenerate Physics Features"):
#     current_df = st.session_state.get('race_data', pd.DataFrame())
#     
#     if not current_df.empty:
#         fixed_df = regenerate_sim_history(current_df)
#         
#         st.session_state['race_data'] = fixed_df
#         fixed_df.to_csv("Race_Data_with_Physics.csv", index=False)
#         st.write("Saved to 'Race_Data_with_Physics.csv'. Please use this file.")

# ---------------------------------------------------------
# 🔍 FULL ML DIAGNOSTIC SUITE (MANUAL + CACHED + LIGHT UI)
# ---------------------------------------------------------

# 1) Manual trigger
st.markdown("---")
if st.button("Run ML Diagnostic Suite"):
    st.session_state.run_ml_diag = True

# 2) Cached heavy computation
@st.cache_data
def compute_ml_diagnostics(history_df: pd.DataFrame):
    """
    Heavy ML diagnostic computation.
    Runs only when explicitly triggered, then cached.
    """
    diag = {
        "training_samples": 0,
        "feature_info": None,
        "ohe_categories": None,
        "raw_shape": None,
        "transformed_shape": None,
        "live_row": None,
        "ml_probs": None,
        "sim_probs": None,
        "disagreement": None,
        "blend_weight": None,
        "brier": None,
        "expected_regret": None,
        "chaos": None,
    }

    model, n_samples = get_trained_model(history_df)
    diag["training_samples"] = n_samples

    if model is None:
        return diag  # nothing else to do

    # --- Feature / transformer info ---
    pre = model.named_steps["pre"]
    clf = model.named_steps["clf"]

    cat_features = pre.transformers_[0][2]
    num_features = pre.transformers_[1][2]
    ohe = pre.named_transformers_["cat"]

    diag["feature_info"] = {
        "categorical": cat_features,
        "numeric": num_features,
    }
    diag["ohe_categories"] = ohe.categories_

    # --- Transformed shape (debug on last 200 modern rows) ---
    X_debug, _, _, _ = build_training_data(history_df.tail(200))
    if X_debug is not None and not X_debug.empty:
        transformed = pre.transform(X_debug.head(1))
        diag["raw_shape"] = X_debug.head(1).shape
        diag["transformed_shape"] = transformed.shape

    # --- Live prediction diagnostics on the latest race ---
    last = history_df.tail(1).iloc[0]
    v1, v2, v3 = last["vehicle_1"], last["vehicle_2"], last["vehicle_3"]
    lane = last["lane"]
    k_idx = int(lane.split(" ")[1]) - 1
    k_type = last[f"lap_{k_idx+1}_track"]

    live_row = build_single_feature_row(
        v1, v2, v3, k_idx, k_type, history_df,
        user_vehicle_priors=None,
        sim_meta_live=None,
    )
    diag["live_row"] = live_row

    # ML prediction (probabilities in 0–1)
    ml_probs = model.predict_proba(live_row)[0]
    ml_probs_dict = {
        v1: float(ml_probs[0]),
        v2: float(ml_probs[1]),
        v3: float(ml_probs[2]),
    }
    diag["ml_probs"] = ml_probs_dict

    # SIM prediction (0–1 probs expected here)
    sim_probs = compute_sim_probs(v1, v2, v3, history_df)
    diag["sim_probs"] = sim_probs

    # Disagreement info
    ml_top = max(ml_probs)
    sim_top = max(sim_probs.values())
    ml_winner = [v1, v2, v3][int(ml_probs.argmax())]
    sim_winner = max(sim_probs, key=sim_probs.get)

    diag["disagreement"] = {
        "ml_top": ml_top,
        "sim_top": sim_top,
        "ml_winner": ml_winner,
        "sim_winner": sim_winner,
    }

    # Blend diagnostics (if you still want this simple view)
    blend_weight = compute_blend_weight(sim_top, ml_top)
    diag["blend_weight"] = blend_weight

    # Brier scores (SIM + ML vs actual winner)
    actual_winner = last["actual_winner"]
    diag["brier"] = {
        "sim_brier": compute_brier(sim_probs, actual_winner),
        "ml_brier": compute_brier(ml_probs_dict, actual_winner),
    }

    # Expected regret (whatever your implementation is)
    diag["expected_regret"] = compute_expected_regret(sim_probs, ml_probs)

    # Chaos condition
    chaos = (
        sim_top > 0.70
        and ml_top > 0.70
        and sim_winner != ml_winner
    )
    diag["chaos"] = chaos

    return diag


# 3) Render (only when triggered) in an expander
if st.session_state.get("run_ml_diag", False):
    with st.expander("🔧 Full ML Diagnostic Suite", expanded=True):
        try:
            diag = compute_ml_diagnostics(history)

            st.write("## 📌 ML Training Diagnostics")
            st.write("Training samples:", diag["training_samples"])

            if diag["training_samples"] == 0:
                st.warning("ML model not trained (not enough usable samples).")
            else:
                # Feature info
                st.write("### 🧩 Feature Columns")
                st.write("Categorical:", diag["feature_info"]["categorical"])
                st.write("Numeric:", diag["feature_info"]["numeric"])

                st.write("### 🔠 OneHotEncoder Categories")
                st.write(diag["ohe_categories"])

                # Shapes
                if diag["raw_shape"] is not None:
                    st.write("### 🧮 Transformed Feature Vector Shape")
                    st.write("Raw shape:", diag["raw_shape"])
                    st.write("Transformed shape:", diag["transformed_shape"])

                st.write("## 🚦 Live Prediction Diagnostics")

                st.write("### 🧪 Live Feature Row")
                st.write(diag["live_row"])

                st.write("### 🤖 ML Probabilities")
                st.write(diag["ml_probs"])

                st.write("### 🏎️ SIM Probabilities")
                st.write(diag["sim_probs"])

                st.write("### ⚔️ ML vs SIM Disagreement")
                st.write(diag["disagreement"])

                st.write("## 🔀 Blend Diagnostics")
                st.write("Blend Weight (ML share):", diag["blend_weight"])

                st.write("### 📉 Brier Scores")
                st.write(diag["brier"])

                st.write("### 😬 Expected Regret")
                st.write(diag["expected_regret"])

                st.write("### 🌪️ Chaos Mode Triggered:", diag["chaos"])

        except Exception as e:
            st.error(f"ML Debug Error: {e}")
