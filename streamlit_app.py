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

# --- 2. DATA ARCHITECT (FULL PODIUM SCHEMA) ---
def load_and_migrate_data():
    # Full data structure to capture MT, Motor, Supercar / Desert, Dirt, Dirt / SC, Motor, MT
    cols = ['V1', 'V2', 'V3', 
            'Lap_1_Track', 'Lap_1_Len', 'Lap_2_Track', 'Lap_2_Len', 'Lap_3_Track', 'Lap_3_Len', 
            'Rank_1', 'Rank_2', 'Rank_3', 'Predicted']
    
    if not os.path.exists(CSV_FILE):
        return pd.DataFrame(columns=cols)
    
    try:
        df = pd.read_csv(CSV_FILE)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Ensure all columns exist for the new Podium Model
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        
        # Numeric cleanup for ML calculations
        for i in range(1, 4):
            l_col = f'Lap_{i}_Len'
            df[l_col] = pd.to_numeric(df[l_col], errors='coerce').fillna(33.3)
            
        return df
    except Exception:
        return pd.DataFrame(columns=cols)

history = load_and_migrate_data()

# --- 3. THE COMPLETE ML ENGINE (PODIUM & CHAIN LOGIC) ---
def run_master_simulation(v1, v2, v3, k_idx, k_type, history_df, iterations=5000):
    vehicles = [v1, v2, v3]
    lap_probs = {0: None, 1: None, 2: None}
    avg_lens = [33.3, 33.3, 33.4]
    std_lens = [8.0, 8.0, 8.0]
    
    # ML PHASE 1: PODIUM PERFORMANCE INDEX (VPI)
    # The AI learns not just from winners, but from who almost won
    vpi = {v: 1.0 for v in vehicles}
    if not history_df.empty and 'Rank_1' in history_df.columns:
        for v in vehicles:
            r1_count = len(history_df[history_df['Rank_1'] == v])
            r2_count = len(history_df[history_df['Rank_2'] == v])
            # 1st = 1% speed boost, 2nd = 0.4% speed boost in simulations
            vpi[v] = 1.0 + (r1_count * 0.01) + (r2_count * 0.004)

    # ML PHASE 2: MARKOV CHAIN PATTERNS (Any sequence)
    if not history_df.empty:
        anchor_col = f"Lap_{k_idx + 1}_Track"
        matches = history_df[history_df[anchor_col] == k_type].tail(50)
        if not matches.empty:
            for i in range(3):
                if i != k_idx:
                    t_col = f"Lap_{i+1}_Track"
                    counts = matches[t_col].value_counts(normalize=True)
                    probs = counts.reindex(TRACK_OPTIONS, fill_value=0).values
                    if probs.sum() > 0: lap_probs[i] = probs / probs.sum()

    # MONTE CARLO CORE
    sim_terrains, sim_lengths = [], []
    for i in range(3):
        if i == k_idx:
            sim_terrains.append(np.full(iterations, k_type))
        else:
            p = lap_probs[i] if lap_probs[i] is not None else None
            sim_terrains.append(np.random.choice(TRACK_OPTIONS, size=iterations, p=p))
        sim_lengths.append(np.random.normal(avg_lens[i], std_lens[i], iterations))

    # Length Normalization (100% Rule)
    len_matrix = np.column_stack(sim_lengths)
    len_matrix = np.clip(len_matrix, 5, 90)
    len_matrix = (len_matrix.T / len_matrix.sum(axis=1)).T 
    
    terrain_matrix = np.column_stack(sim_terrains)
    results = {}
    for v in vehicles:
        base_speed = np.vectorize(SPEED_DATA[v].get)(terrain_matrix)
        final_speed = base_speed * vpi[v] * np.random.normal(1.0, 0.02, (iterations, 3))
        results[v] = np.sum(len_matrix / final_speed, axis=1)

    # Convert to Win %
    winners = np.argmin(np.array([results[v] for v in vehicles]), axis=0)
    win_pcts = pd.Series(winners).value_counts(normalize=True).sort_index() * 100
    return {vehicles[i]: win_pcts.get(i, 0) for i in range(3)}, vpi

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("🚦 Race Setup")
    lap_options = {"Lap 1": 0, "Lap 2": 1, "Lap 3": 2}
    slot_name = st.selectbox("Visible Position", list(lap_options.keys()))
    k_idx, k_type = lap_options[slot_name], st.selectbox("Visible Track", TRACK_OPTIONS)
    
    st.divider()
    # Unique Vehicle Selection
    v1_sel = st.selectbox("Vehicle 1", ALL_VEHICLES, index=ALL_VEHICLES.index("Supercar"))
    v2_sel = st.selectbox("Vehicle 2", [v for v in ALL_VEHICLES if v != v1_sel], index=0)
    v3_sel = st.selectbox("Vehicle 3", [v for v in ALL_VEHICLES if v not in [v1_sel, v2_sel]], index=0)
    
    if st.button("🚀 PREDICT", type="primary", use_container_width=True):
        probs, vpi_res = run_master_simulation(v1_sel, v2_sel, v3_sel, k_idx, k_type, history)
        st.session_state['master'] = {'p': probs, 'vpi': vpi_res, 'ctx': {'v': [v1_sel, v2_sel, v3_sel], 'idx': k_idx, 't': k_type}}

# --- 5. DASHBOARD ---
st.title("🏁 AI RACE MASTER: PODIUM EDITION")

# Win Accuracy Metric
if not history.empty and 'Rank_1' in history.columns and 'Predicted' in history.columns:
    valid = history.dropna(subset=['Rank_1', 'Predicted'])
    if not valid.empty:
        acc = (valid['Predicted'] == valid['Rank_1']).mean() * 100
        st.metric("🎯 Prediction Accuracy (Win)", f"{acc:.1f}%")

if 'master' in st.session_state:
    res = st.session_state['master']
    m_grid = grid(3, vertical_align="center")
    for v, val in res['p'].items():
        m_grid.metric(v, f"{val:.1f}%")

# --- 6. TELEMETRY (THE PODIUM LOG) ---
st.divider()
st.subheader("📝 POST-RACE PODIUM REPORT")
ctx = st.session_state.get('master', {'ctx': {'v': [v1_sel, v2_sel, v3_sel], 'idx':0, 't': TRACK_OPTIONS[0]}})['ctx']



with st.form("tele_form"):
    st.write("Record the EXACT finishing order and lengths:")
    c_r1, c_r2, c_r3 = st.columns(3)
    with c_r1: r1 = st.selectbox("🏆 1st Place", ctx['v'], index=0)
    with c_r2: r2 = st.selectbox("🥈 2nd Place", [v for v in ctx['v'] if v != r1], index=0)
    with c_r3: r3 = st.selectbox("🥉 3rd Place", [v for v in ctx['v'] if v not in [r1, r2]], index=0)
    
    st.write("--- Track Geometry ---")
    c_a, c_b, c_c = st.columns(3)
    with c_a:
        s1t = st.selectbox("L1 Track", TRACK_OPTIONS, index=TRACK_OPTIONS.index(ctx['t']) if ctx['idx']==0 else 0)
        s1l = st.number_input("L1 Length %", 1, 100, 33)
    with c_b:
        s2t = st.selectbox("L2 Track", TRACK_OPTIONS, index=TRACK_OPTIONS.index(ctx['t']) if ctx['idx']==1 else 0)
        s2l = st.number_input("L2 Length %", 1, 100, 33)
    with c_c:
        s3t = st.selectbox("L3 Track", TRACK_OPTIONS, index=TRACK_OPTIONS.index(ctx['t']) if ctx['idx']==2 else 0)
        s3l = st.number_input("L3 Length %", 1, 100, 34)

    if st.form_submit_button("💾 SAVE FULL PODIUM REPORT"):
        if s1l + s2l + s3l != 100:
            st.error("❌ Total must be 100%")
        else:
            p_val = max(st.session_state['master']['p'], key=st.session_state['master']['p'].get) if 'master' in st.session_state else "N/A"
            row = {'V1': ctx['v'][0], 'V2': ctx['v'][1], 'V3': ctx['v'][2],
                   'Lap_1_Track': s1t, 'Lap_1_Len': s1l, 'Lap_2_Track': s2t, 'Lap_2_Len': s2l,
                   'Lap_3_Track': s3t, 'Lap_3_Len': s3l, 
                   'Predicted': p_val, 'Rank_1': r1, 'Rank_2': r2, 'Rank_3': r3}
            pd.concat([history, pd.DataFrame([row])], ignore_index=True).to_csv(CSV_FILE, index=False)
            st.success("Podium saved and AI trained!")
            st.rerun()

# --- 7. ANALYTICS ---
if not history.empty:
    st.divider()
    st.header("📊 Deep Intelligence Analytics")
    t1, t2, t3 = st.tabs(["🧠 Sequence Brain", "🧬 Vehicle Stats", "📂 Database"])
    
    with t1:
        st.write("### Track Transition Probability")
        if 'Lap_1_Track' in history.columns:
            m = pd.crosstab(history['Lap_1_Track'], history['Lap_2_Track'], normalize='index') * 100
            st.dataframe(m.style.format("{:.0f}%").background_gradient(cmap="Blues", axis=1))
            
    with t2:
        st.write("### Performance Index (Based on Podium Points)")
        # Calculate points: 1st=10, 2nd=5, 3rd=0
        points = {}
        for _, row in history.iterrows():
            points[row['Rank_1']] = points.get(row['Rank_1'], 0) + 10
            points[row['Rank_2']] = points.get(row['Rank_2'], 0) + 5
        st.write(pd.DataFrame.from_dict(points, orient='index', columns=['Total Points']).sort_values('Total Points', ascending=False))

    with t3:
        st.dataframe(history.sort_index(ascending=False), use_container_width=True)
