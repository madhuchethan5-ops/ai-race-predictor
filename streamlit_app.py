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
        return df
    except Exception:
        return pd.DataFrame(columns=cols)

history = load_and_migrate_data()

# --- 3. THE ML ENGINE (NEXT-GEN, SAME API) ---
def run_simulation(
    v1, v2, v3,
    k_idx,           # known lap index: 0, 1, or 2
    k_type,          # known track type for that lap
    history_df,
    iterations=5000,
    alpha_prior=1.0, # Bayesian prior wins
    beta_prior=1.0,  # Bayesian prior losses
    smoothing=0.5,   # Markov/Dirichlet smoothing
    base_len_mean=33.3,
    base_len_std=5.0,
    calib_min_hist=50 # minimum rows for temperature calibration
):
    vehicles = [v1, v2, v3]

    # 1. BAYESIAN REINFORCEMENT (vehicle performance index, VPI)
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

    # 2. LEARNED GEOMETRY / TRACK-SPECIFIC LENGTH DISTRIBUTIONS
    def learned_length_dist(lap_idx, track_type):
        if history_df.empty:
            return base_len_mean, base_len_std

        t_col = f"Lap_{lap_idx + 1}_Track"
        l_col = f"Lap_{lap_idx + 1}_Len"
        if t_col not in history_df.columns or l_col not in history_df.columns:
            return base_len_mean, base_len_std

        mask = (history_df[t_col] == track_type)
        subset = history_df[mask]
        if subset.empty:
            return base_len_mean, base_len_std

        vals = pd.to_numeric(subset[l_col], errors='coerce').dropna()
        if len(vals) < 5:
            return base_len_mean, base_len_std

        mu = vals.mean()
        sigma = vals.std(ddof=1)
        if not np.isfinite(mu) or sigma <= 0:
            return base_len_mean, base_len_std

        return float(mu), float(sigma)

    # 3. SMOOTHED MARKOV TRANSITIONS FOR HIDDEN LAPS
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

    # 4. SAMPLE TERRAIN AND LENGTHS WITH LEARNED GEOMETRY
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
            mu, sigma = learned_length_dist(i, t)
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

    # 5. BETTER NOISE MODELING FOR VEHICLE SPEEDS
    def sample_vehicle_times(vehicle):
        base_speed = np.vectorize(SPEED_DATA[vehicle].get)(terrain_matrix)
        base_speed = np.clip(base_speed, 0.1, None)

        speed_std = 0.03 * base_speed

        veh_factor = np.random.normal(1.0, 0.03, size=(iterations, 1))
        lap_factor = np.random.normal(1.0, 0.02, size=(iterations, 3))

        effective_speed = base_speed * veh_factor * lap_factor
        effective_speed = np.clip(effective_speed, 0.1, None)

        return np.sum(len_matrix / (effective_speed * vpi[vehicle]), axis=1)

    results = {v: sample_vehicle_times(v) for v in vehicles}

    # 6. RAW MONTE CARLO WIN PROBABILITIES
    total_times = np.vstack([results[v] for v in vehicles])
    winners = np.argmin(total_times, axis=0)
    freq = pd.Series(winners).value_counts(normalize=True).sort_index()
    raw_probs = np.array([freq.get(i, 0.0) for i in range(3)], dtype=float)
    raw_probs = np.clip(raw_probs, 1e-6, 1.0)
    raw_probs /= raw_probs.sum()

    # 7. SIMPLE CALIBRATION (TEMPERATURE SCALING)
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
st.subheader("📝 POST-RACE REPORT (Revealed slot is locked)")

default_ctx = {'v': [v1_sel, v2_sel, v3_sel], 'idx': 0, 't': TRACK_OPTIONS[0], 'slot': 'Lap 1'}
ctx = st.session_state.get('res', {'ctx': default_ctx})['ctx']
current_slot = ctx.get('slot', 'Lap 1')

with st.form("tele_form"):
    winner = st.selectbox("🏆 Actual Winner", ctx['v'])
    
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

    if st.form_submit_button("💾 SAVE & TRAIN"):
        if s1l + s2l + s3l != 100:
            st.error("❌ Total must be 100%")
        else:
            if 'res' in st.session_state:
                probs_now = st.session_state['res']['p']
                p_val = max(probs_now, key=probs_now.get)
                top_prob = probs_now[p_val] / 100.0
            else:
                p_val = "N/A"
                top_prob = np.nan

            was_correct = (p_val == winner) if p_val != "N/A" else np.nan

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
            st.toast("AI Learned and Lane Context Saved!", icon="🧠")
            st.rerun()

# --- 7. ANALYTICS (LANE TRACKER & ML BRAIN) ---
if not history.empty:
    st.divider()
    t1, t2, t3 = st.tabs(["🧠 ML Pattern Brain", "🚦 Lane Tracker", "📂 History"])
    
    with t1:
        st.write("### Track Transition Matrix")
        if 'Lap_1_Track' in history.columns and 'Lap_2_Track' in history.columns:
            m = pd.crosstab(history['Lap_1_Track'], history['Lap_2_Track'], normalize='index') * 100
            st.dataframe(m.style.format("{:.0f}%").background_gradient(cmap="Blues", axis=1))
    
    with t2:
        st.write("### Win Rate by Lane Context")
        if 'Lane' in history.columns and history['Lane'].notna().any():
            lane_stats = pd.crosstab(history['Lane'], history['Actual_Winner'], normalize='index') * 100
            st.dataframe(lane_stats.style.format("{:.1f}%").background_gradient(cmap="YlOrRd", axis=1))
        else:
            st.info("Record more races to see Lane win rates.")
            
    with t3: 
        st.dataframe(history.sort_index(ascending=False), use_container_width=True)
