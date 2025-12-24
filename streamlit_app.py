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

# --- 2. DATA ARCHITECT (FIXED MAPPING) ---
def load_and_migrate_data():
    # Standard structure to save all 3 vehicles and all 3 tracks in one row
    cols = ['Vehicle_1', 'Vehicle_2', 'Vehicle_3', 
            'Lap_1_Track', 'Lap_1_Len', 'Lap_2_Track', 'Lap_2_Len', 'Lap_3_Track', 'Lap_3_Len', 
            'Actual_Winner', 'Predicted_Winner']
    
    if not os.path.exists(CSV_FILE):
        return pd.DataFrame(columns=cols)
    
    try:
        df = pd.read_csv(CSV_FILE)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Mapping old versions to current structure
        rename_map = {
            'V1': 'Vehicle_1', 'V2': 'Vehicle_2', 'V3': 'Vehicle_3',
            'Rank_1': 'Actual_Winner', 'Predicted': 'Predicted_Winner',
            'Stage_1_Track': 'Lap_1_Track', 'Stage_2_Track': 'Lap_2_Track', 'Stage_3_Track': 'Lap_3_Track'
        }
        df = df.rename(columns=rename_map)
        
        for i in range(1, 4):
            l_col = f'Lap_{i}_Len'
            if l_col in df.columns:
                df[l_col] = pd.to_numeric(df[l_col], errors='coerce').fillna(33.3)
                
        return df
    except Exception:
        return pd.DataFrame(columns=cols)

history = load_and_migrate_data()

# --- 3. THE ML ENGINE (TRACK CHAIN + WINNER BOOST) ---
def run_simulation(v1, v2, v3, k_idx, k_type, history_df, iterations=5000):
    vehicles = [v1, v2, v3]
    lap_probs = {0: None, 1: None, 2: None}
    avg_lens = [33.3, 33.3, 33.4]
    std_lens = [8.0, 8.0, 8.0]
    
    # ML PHASE 1: Winner-Only Performance Index
    vpi = {v: 1.0 for v in vehicles}
    if not history_df.empty and 'Actual_Winner' in history_df.columns:
        wins = history_df['Actual_Winner'].value_counts()
        for v in vehicles:
            vpi[v] = 1.0 + (wins.get(v, 0) * 0.005) # 0.5% boost per win

    # ML PHASE 2: Track Pattern Recognition
    if not history_df.empty:
        filter_col = f"Lap_{k_idx + 1}_Track"
        matches = history_df[history_df[filter_col] == k_type].tail(50)
        if not matches.empty:
            for i in range(3):
                if i != k_idx:
                    t_col = f"Lap_{i+1}_Track"
                    if t_col in matches.columns:
                        counts = matches[t_col].value_counts(normalize=True)
                        probs = counts.reindex(TRACK_OPTIONS, fill_value=0).values
                        if probs.sum() > 0: lap_probs[i] = probs / probs.sum()

    # MONTE CARLO
    sim_terrains, sim_lengths = [], []
    for i in range(3):
        if i == k_idx: sim_terrains.append(np.full(iterations, k_type))
        else:
            p = lap_probs[i] if lap_probs[i] is not None else None
            sim_terrains.append(np.random.choice(TRACK_OPTIONS, size=iterations, p=p))
        sim_lengths.append(np.random.normal(avg_lens[i], std_lens[i], iterations))

    # Normalize Lengths to 100%
    len_matrix = np.column_stack(sim_lengths)
    len_matrix = np.clip(len_matrix, 5, 90)
    len_matrix = (len_matrix.T / len_matrix.sum(axis=1)).T 
    
    terrain_matrix = np.column_stack(sim_terrains)
    results = {}
    for v in vehicles:
        base_speed = np.vectorize(SPEED_DATA[v].get)(terrain_matrix)
        final_speed = base_speed * vpi[v] * np.random.normal(1.0, 0.02, (iterations, 3))
        results[v] = np.sum(len_matrix / final_speed, axis=1)

    winners = np.argmin(np.array([results[v] for v in vehicles]), axis=0)
    win_pcts = pd.Series(winners).value_counts(normalize=True).sort_index() * 100
    return {vehicles[i]: win_pcts.get(i, 0) for i in range(3)}, vpi

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("🚦 Pre-Race Setup")
    lap_options = {"Lap 1": 0, "Lap 2": 1, "Lap 3": 2}
    slot_name = st.selectbox("Visible Position", list(lap_options.keys()))
    k_idx, k_type = lap_options[slot_name], st.selectbox("Visible Track", TRACK_OPTIONS)
    
    st.divider()
    v1_sel = st.selectbox("Vehicle 1", ALL_VEHICLES, index=ALL_VEHICLES.index("Supercar"))
    v2_sel = st.selectbox("Vehicle 2", [v for v in ALL_VEHICLES if v != v1_sel], index=0)
    v3_sel = st.selectbox("Vehicle 3", [v for v in ALL_VEHICLES if v not in [v1_sel, v2_sel]], index=0)
    
    if st.button("🚀 PREDICT", type="primary", use_container_width=True):
        probs, vpi_res = run_simulation(v1_sel, v2_sel, v3_sel, k_idx, k_type, history)
        st.session_state['res'] = {'p': probs, 'vpi': vpi_res, 'ctx': {'v': [v1_sel, v2_sel, v3_sel], 'idx': k_idx, 't': k_type}}

# --- 5. DASHBOARD ---
st.title("🏁 AI RACE MASTER: WINNER EDITION")

# Accuracy Tracker
if not history.empty and 'Actual_Winner' in history.columns and 'Predicted_Winner' in history.columns:
    valid = history.dropna(subset=['Actual_Winner', 'Predicted_Winner'])
    if not valid.empty:
        acc = (valid['Predicted_Winner'] == valid['Actual_Winner']).mean() * 100
        st.metric("🎯 Prediction Accuracy", f"{acc:.1f}%")

if 'res' in st.session_state:
    res = st.session_state['res']
    m_grid = grid(3, vertical_align="center")
    for v, val in res['p'].items():
        m_grid.metric(v, f"{val:.1f}%")

# --- 6. TELEMETRY (POST-RACE WINNER LOG) ---
st.divider()
st.subheader("📝 POST-RACE REPORT")
ctx = st.session_state.get('res', {'ctx': {'v': [v1_sel, v2_sel, v3_sel], 'idx':0, 't': TRACK_OPTIONS[0]}})['ctx']

with st.form("tele_form"):
    winner = st.selectbox("🏆 Actual Winner", ctx['v'])
    
    st.write("--- Track Geometry (Revealed during race) ---")
    c_a, c_b, c_c = st.columns(3)
    with c_a:
        s1t = st.selectbox("Lap 1 Track", TRACK_OPTIONS, index=TRACK_OPTIONS.index(ctx['t']) if ctx['idx']==0 else 0)
        s1l = st.number_input("Lap 1 %", 1, 100, 33)
    with c_b:
        s2t = st.selectbox("Lap 2 Track", TRACK_OPTIONS, index=TRACK_OPTIONS.index(ctx['t']) if ctx['idx']==1 else 0)
        s2l = st.number_input("Lap 2 %", 1, 100, 33)
    with c_c:
        s3t = st.selectbox("Lap 3 Track", TRACK_OPTIONS, index=TRACK_OPTIONS.index(ctx['t']) if ctx['idx']==2 else 0)
        s3l = st.number_input("Lap 3 %", 1, 100, 34)

    if st.form_submit_button("💾 SAVE RACE RESULTS"):
        if s1l + s2l + s3l != 100:
            st.error("❌ Laps must total 100%")
        else:
            p_val = max(st.session_state['res']['p'], key=st.session_state['res']['p'].get) if 'res' in st.session_state else "N/A"
            row = {'Vehicle_1': ctx['v'][0], 'Vehicle_2': ctx['v'][1], 'Vehicle_3': ctx['v'][2],
                   'Lap_1_Track': s1t, 'Lap_1_Len': s1l, 'Lap_2_Track': s2t, 'Lap_2_Len': s2l,
                   'Lap_3_Track': s3t, 'Lap_3_Len': s3l, 
                   'Predicted_Winner': p_val, 'Actual_Winner': winner}
            pd.concat([history, pd.DataFrame([row])], ignore_index=True).to_csv(CSV_FILE, index=False)
            st.toast("Race recorded and AI updated!", icon="🧠")
            st.rerun()

# --- 7. ANALYTICS ---
if not history.empty:
    st.divider()
    t1, t2 = st.tabs(["🧠 Track Patterns", "📂 Database"])
    with t1:
        st.write("### Track Transition Probability Matrix")
        
        if 'Lap_1_Track' in history.columns:
            m = pd.crosstab(history['Lap_1_Track'], history['Lap_2_Track'], normalize='index') * 100
            st.dataframe(m.style.format("{:.0f}%").background_gradient(cmap="Blues", axis=1))
    with t2:
        st.dataframe(history.sort_index(ascending=False), use_container_width=True)
