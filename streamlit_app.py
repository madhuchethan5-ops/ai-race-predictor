import streamlit as st
import pandas as pd
import numpy as np
import os
from streamlit_extras.grid import grid

# --- 1. CORE PHYSICS CONFIGURATION ---
SPEED_DATA = {
    "Monster Truck": {"Expressway": 110, "Desert": 55, "Dirt": 81, "Potholes": 48, "Bumpy": 75, "Highway": 100},
    "ORV":           {"Expressway": 140, "Desert": 57, "Dirt": 92, "Potholes": 49, "Bumpy": 76, "Highway": 112},
    "Motorcycle":    {"Expressway": 94,  "Desert": 45, "Dirt": 76, "Potholes": 36, "Bumpy": 66, "Highway": 89},
    "Stock Car":     {"Expressway": 100, "Desert": 50, "Dirt": 80, "Potholes": 45, "Bumpy": 72, "Highway": 99},
    "SUV":           {"Expressway": 180, "Desert": 63, "Dirt": 100,"Potholes": 60, "Bumpy": 80, "Highway": 143},
    "Car":           {"Expressway": 235, "Desert": 70, "Dirt": 120,"Potholes": 68, "Bumpy": 81, "Highway": 180},
    "ATV":           {"Expressway": 80,  "Desert": 40, "Dirt": 66, "Potholes": 32, "Bumpy": 60, "Highway": 80},
    "Sports Car":    {"Expressway": 300, "Desert": 72, "Dirt": 130,"Potholes": 72, "Bumpy": 91, "Highway": 240},
    "Supercar":      {"Expressway": 390, "Desert": 80, "Dirt": 134,"Potholes": 77, "Bumpy": 99, "Highway": 320},
}

ALL_VEHICLES = sorted(list(SPEED_DATA.keys()))
TRACK_OPTIONS = sorted(list(SPEED_DATA["Car"].keys()))
CSV_FILE = 'race_history.csv'

st.set_page_config(layout="wide", page_title="AI Race Master Pro", page_icon="🏎️")

# === DATA QUALITY & AUTO-CLEANING ===

VALID_TRACKS = set(TRACK_OPTIONS)

TRACK_ALIASES = {
    "Road": "Highway",
    "road": "Highway",
    "Normal road": "Highway",
    "normal road": "Highway",
    "Normal": "Highway",
}

def auto_clean_history(df: pd.DataFrame):
    """Automatically clean race history: fix track names, lanes, invalid rows, normalize lap lengths."""
    if df.empty:
        return df, []

    issues = []
    df = df.copy()

    # Fix Lane values
    lane_map = {"1": "Lap 1", "2": "Lap 2", "3": "Lap 3", 1: "Lap 1", 2: "Lap 2", 3: "Lap 3"}
    if "Lane" in df.columns:
        df["Lane"] = df["Lane"].replace(lane_map)

    # Fix track names with aliases and fallback to mode
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

    # Enforce numeric lap lengths
    for lap in [1, 2, 3]:
        col = f"Lap_{lap}_Len"
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalize lengths if they don't sum to 100
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

# === METRICS & ANALYTICS HELPERS ===

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

def compute_learned_geometry(history: pd.DataFrame):
    """Mean/std of Lap lengths per track, across all laps (because lengths vary wildly)."""
    if history.empty:
        return None
    rows = []
    for lap in [1, 2, 3]:
        t_col = f"Lap_{lap}_Track"
        l_col = f"Lap_{lap}_Len"
        if t_col not in history.columns or l_col not in history.columns:
            continue
        tmp = history[[t_col, l_col]].copy()
        tmp[l_col] = pd.to_numeric(tmp[l_col], errors='coerce')
        tmp = tmp.dropna()
        if tmp.empty:
            continue
        g = tmp.groupby(t_col)[l_col].agg(['mean', 'std', 'count']).reset_index()
        g['Lap'] = lap
        rows.append(g)
    if not rows:
        return None
    geom = pd.concat(rows, ignore_index=True)
    geom = geom.rename(columns={geom.columns[0]: 'Track'})
    return geom[['Lap', 'Track', 'mean', 'std', 'count']]

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

# --- 2. DATA ARCHITECT (FIXED & HARDENED) ---
def load_and_migrate_data():
    cols = ['Vehicle_1', 'Vehicle_2', 'Vehicle_3',
            'Lap_1_Track', 'Lap_1_Len',
            'Lap_2_Track', 'Lap_2_Len',
            'Lap_3_Track', 'Lap_3_Len',
            'Actual_Winner', 'Predicted_Winner',
            'Lane', 'Top_Prob', 'Was_Correct']

    if not os.path.exists(CSV_FILE):
        return pd.DataFrame(columns=cols)

    try:
        df = pd.read_csv(CSV_FILE)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        rename_map = {
            'V1': 'Vehicle_1', 'V2': 'Vehicle_2', 'V3': 'Vehicle_3',
            'Visible_Track': 'Lap_1_Track', 'Visible_Lane_Length (%)': 'Lap_1_Len',
            'Hidden_1': 'Lap_2_Track', 'Hidden_1_Len': 'Lap_2_Len',
            'Hidden_2': 'Lap_3_Track', 'Hidden_2_Len': 'Lap_3_Len'
        }
        df = df.rename(columns=rename_map)

        for c in cols:
            if c not in df.columns:
                df[c] = np.nan

        df = df.replace('None', np.nan)

        df, issues = auto_clean_history(df)
        st.session_state["data_quality_issues"] = issues

        return df

    except Exception:
        return pd.DataFrame(columns=cols)

history = load_and_migrate_data()

# --- 3. THE ML ENGINE (NEXT-GEN, UPGRADED GEOMETRY, SAME API) ---
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
    base_len_std=15.0,   # wider default because lengths are very random
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

    # 2. UPGRADED GEOMETRY: track-specific, across all laps
    def learned_length_dist(track_type):
        if history_df.empty:
            return base_len_mean, base_len_std

        vals_all = []
        for lap in [1, 2, 3]:
            t_col = f"Lap_{lap}_Track"
            l_col = f"Lap_{lap}_Len"
            if t_col not in history_df.columns or l_col not in history_df.columns:
                continue
            mask = (history_df[t_col] == track_type)
            subset = history_df.loc[mask, l_col]
            subset = pd.to_numeric(subset, errors='coerce').dropna()
            if not subset.empty:
                vals_all.append(subset)
        if not vals_all:
            return base_len_mean, base_len_std

        vals = pd.concat(vals_all)
        if len(vals) < 5:
            return base_len_mean, base_len_std

        mu = vals.mean()
        sigma = vals.std(ddof=1)
        if not np.isfinite(mu) or sigma <= 0:
            return base_len_mean, base_len_std

        return float(mu), float(sigma)

    # 3. SMOOTHED MARKOV TRANSITIONS
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
                to_col   = f"Lap_{j + 1}_Track"
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
                t_col = f"Lap_{j+1}_Track"
                if t_col in matches.columns and not matches.empty:
                    counts = matches[t_col].value_counts()
                    arr = counts.reindex(TRACK_OPTIONS, fill_value=0).astype(float)
                    arr = arr + smoothing
                    probs = arr / arr.sum()
                    lap_probs[j] = probs.values
                if lap_probs[j] is None and j in global_transitions:
                    lap_probs[j] = global_transitions[j].values

    # 4. SAMPLE TERRAIN AND LENGTHS
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

    # 5. NOISE MODELING
    def sample_vehicle_times(vehicle):
        base_speed = np.vectorize(SPEED_DATA[vehicle].get)(terrain_matrix)
        base_speed = np.clip(base_speed, 0.1, None)

        veh_factor = np.random.normal(1.0, 0.03, size=(iterations, 1))
        lap_factor = np.random.normal(1.0, 0.02, size=(iterations, 3))

        effective_speed = base_speed * veh_factor * lap_factor
        effective_speed = np.clip(effective_speed, 0.1, None)

        return np.sum(len_matrix / (effective_speed * vpi[vehicle]), axis=1)

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
        avg_acc  = recent['Was_Correct'].mean()
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

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("🚦 Race Setup")
    lap_map = {"Lap 1": 0, "Lap 2": 1, "Lap 3": 2}
    slot_name = st.selectbox("Revealed Slot", list(lap_map.keys()))
    k_idx, k_type = lap_map[slot_name], st.selectbox("Revealed Track", TRACK_OPTIONS)
    
    st.divider()
    v1_sel = st.selectbox("Vehicle 1", ALL_VEHICLES, index=ALL_VEHICLES.index("Supercar"))
    v2_sel = st.selectbox("Vehicle 2", [v for v in ALL_VEHICLES if v != v1_sel], index=0)
    v3_sel = st.selectbox("Vehicle 3", [v for v in ALL_VEHICLES if v not in [v1_sel, v2_sel]], index=0)
    
    if st.button("🚀 PREDICT", type="primary", use_container_width=True):
        probs, vpi_res = run_simulation(v1_sel, v2_sel, v3_sel, k_idx, k_type, history)
        st.session_state['res'] = {
            'p': probs,
            'vpi': vpi_res,
            'ctx': {'v': [v1_sel, v2_sel, v3_sel], 'idx': k_idx, 't': k_type, 'slot': slot_name}
        }

# --- 5. DASHBOARD ---
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

# --- 6. TELEMETRY (LOCKED SLOT & LANE TRACKING) ---
st.divider()
st.subheader("📝 Save Race Report (Revealed slot is locked)")

predicted = st.session_state['res']['p']
predicted_winner = max(predicted, key=predicted.get)

# --- Race Summary Card ---
with st.container():
    st.markdown("### 🧾 Race Summary")
    v1, v2, v3 = ctx['v']
    revealed_lap = ctx['idx'] + 1
    revealed_track = ctx['t']

    st.info(
        f"**Vehicles:** {v1}, {v2}, {v3}\n\n"
        f"**Revealed Lap:** Lap {revealed_lap}\n"
        f"**Revealed Track:** {revealed_track}\n\n"
        f"**Predicted Winner:** {predicted_winner} "
        f"({predicted[predicted_winner]:.1f}%)"
    )

st.caption("Tip: Press **Ctrl + Enter** to save instantly.")

with st.form("tele_form"):
    # Winner selection with auto-highlight
    winner = st.selectbox(
        "🏆 Actual Winner",
        ctx['v'],
        index=ctx['v'].index(st.session_state.get('winner_autofill'))
              if st.session_state.get('winner_autofill') in ctx['v'] else None,
        placeholder=f"Predicted: {predicted_winner}"
    )

    # Quick Fill button
    if st.button("⚡ Quick Fill Predicted Winner"):
        st.session_state['winner_autofill'] = predicted_winner
        st.experimental_rerun()

    # Lap inputs
    c_a, c_b, c_c = st.columns(3)
    with c_a:
        s1t = st.selectbox("L1 Track", TRACK_OPTIONS, index=TRACK_OPTIONS.index(ctx['t']) if ctx['idx']==0 else 0, disabled=(ctx['idx']==0))
        s1l = st.number_input("L1 %", 1, 100, 33)
    with c_b:
        s2t = st.selectbox("L2 Track", TRACK_OPTIONS, index=TRACK_OPTIONS.index(ctx['t']) if ctx['idx']==1 else 0, disabled=(ctx['idx']==1))
        s2l = st.number_input("L2 %", 1, 100, 33)
    with c_c:
        s3t = st.selectbox("L3 Track", TRACK_OPTIONS, index=TRACK_OPTIONS.index(ctx['t']) if ctx['idx']==2 else 0, disabled=(ctx['idx']==2))
        s3l = st.number_input("L3 %", 1, 100, 34)

    save_clicked = st.form_submit_button("💾 SAVE & TRAIN")

    if save_clicked:
        if winner is None:
            st.error("❌ Please select the actual winner before saving.")
            st.stop()

        if s1l + s2l + s3l != 100:
            st.error("❌ Total must be 100%")
            st.stop()

        p_val = predicted_winner
        top_prob = predicted[p_val] / 100.0
        was_correct = (p_val == winner)

        row = {
            'Vehicle_1': ctx['v'][0], 'Vehicle_2': ctx['v'][1], 'Vehicle_3': ctx['v'][2],
            'Lap_1_Track': s1t, 'Lap_1_Len': s1l,
            'Lap_2_Track': s2t, 'Lap_2_Len': s2l,
            'Lap_3_Track': s3t, 'Lap_3_Len': s3l,
            'Predicted_Winner': p_val,
            'Actual_Winner': winner,
            'Lane': current_slot,
            'Top_Prob': top_prob,
            'Was_Correct': was_correct
        }

        pd.concat([history, pd.DataFrame([row])], ignore_index=True).to_csv(CSV_FILE, index=False)
        st.toast("✅ Race saved and AI trained!", icon="🧠")
        st.rerun()
# --- 7. ANALYTICS (MODEL INSIGHTS & BRAIN) ---
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
            c2.metric("Mean Top Probability", f"{metrics['mean_top_prob']*100:.1f}%" if pd.notna(metrics['mean_top_prob']) else "N/A")
            c3.metric("Calibration Error |p̂ - acc|", f"{metrics['calib_error']*100:.2f}%" if pd.notna(metrics['calib_error']) else "N/A")

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

                st.line_chart(
                    calib_table.set_index('Bucket')[['mean_prob', 'emp_acc']],
                    height=300
                )
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

    # 5) VOLATILITY & FEATURE IMPORTANCE (SIMULATED)
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

    # 6) ML PATTERN BRAIN (TRANSITIONS)
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

    # 9) RAW HISTORY
    with tabs[8]:
        st.write("### 📂 Race History")
        st.dataframe(history.sort_index(ascending=False), use_container_width=True)

    # 10) WHAT-IF ANALYSIS PANEL
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
