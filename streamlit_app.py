import streamlit as st
import pandas as pd
import numpy as np
import os
from streamlit_extras.grid import grid

# --- 1. GLOBAL PHYSICS DATA ---
# This is the AI's "Hardcoded" knowledge of vehicle capabilities
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

# --- 2. THE DATA ARCHITECT (MIGRATION & VALIDATION) ---
def load_and_migrate_data():
    """Handles different CSV versions and ensures 'Actual' column exists."""
    if not os.path.exists(CSV_FILE):
        # Create an empty DataFrame with the correct headers immediately
        return pd.DataFrame(columns=['Vehicle_1', 'Vehicle_2', 'Vehicle_3', 
                                   'Lap_1_Track', 'Lap_1_Len', 'Lap_2_Track', 'Lap_2_Len', 
                                   'Lap_3_Track', 'Lap_3_Len', 'Predicted', 'Actual'])
    try:
        df = pd.read_csv(CSV_FILE)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        rename_map = {
            'V1': 'Vehicle_1', 'V2': 'Vehicle_2', 'V3': 'Vehicle_3',
            'Visible_Track': 'Lap_1_Track', 'Visible_Segment_%': 'Lap_1_Len',
            'Hidden_1_Track': 'Lap_2_Track', 'Hidden_2_Track': 'Lap_3_Track'
        }
        df = df.rename(columns=rename_map)
        
        # CRITICAL FIX: Ensure 'Actual' and 'Predicted' columns exist even if file is new
        if 'Actual' not in df.columns:
            df['Actual'] = np.nan
        if 'Predicted' not in df.columns:
            df['Predicted'] = np.nan
            
        for i in range(1, 4):
            l_col = f'Lap_{i}_Len'
            if l_col in df.columns:
                df[l_col] = pd.to_numeric(df[l_col], errors='coerce').fillna(33.3)
        
        return df
    except Exception as e:
        return pd.DataFrame(columns=['Actual', 'Predicted']) # Fallback to prevent crash

# --- 3. THE COMPLETE AI ML ENGINE (BIDIRECTIONAL & SELF-TRAINING) ---
def run_master_simulation(v1, v2, v3, k_idx, k_type, history_df, iterations=5000):
    """
    Main AI Logic:
    1. Adjusts vehicle speeds based on actual win history (VPI).
    2. Predicts hidden tracks using Markov Chain patterns (Forward/Backward).
    3. Simulates unknown lengths based on historical track geometry.
    4. Enforces a strict 100% total race length constraint.
    """
    vehicles = [v1, v2, v3]
    
    # --- ML PART 1: VEHICLE PERFORMANCE INDEX (VPI) ---
    vpi = {v: 1.0 for v in vehicles}
    if not history_df.empty and 'Actual' in history_df.columns:
        win_counts = history_df['Actual'].value_counts()
        for v in vehicles:
            # For every win, AI grants a 0.5% boost to the vehicle's "Driverless" performance
            vpi[v] = 1.0 + (win_counts.get(v, 0) * 0.005)

    # --- ML PART 2: PATTERN RECOGNITION (CHAIN LOGIC) ---
    lap_probs = {0: None, 1: None, 2: None}
    avg_lens = [33.3, 33.3, 33.4]
    std_lens = [8.0, 8.0, 8.0]
    
    if not history_df.empty:
        anchor_col = f"Lap_{k_idx + 1}_Track"
        if anchor_col in history_df.columns:
            matches = history_df[history_df[anchor_col] == k_type].tail(50)
            if not matches.empty:
                for i in range(3):
                    # Predict Hidden Tracks (Forward/Backward)
                    if i != k_idx:
                        t_col = f"Lap_{i+1}_Track"
                        if t_col in matches.columns:
                            counts = matches[t_col].value_counts(normalize=True)
                            probs = counts.reindex(TRACK_OPTIONS, fill_value=0).values
                            if probs.sum() > 0: lap_probs[i] = probs / probs.sum()
                    
                    # Predict Hidden Geometry (Lengths)
                    l_col = f"Lap_{i+1}_Len"
                    if l_col in matches.columns:
                        vals = matches[l_col].dropna()
                        if not vals.empty:
                            avg_lens[i] = vals.mean()
                            if len(vals) > 1: std_lens[i] = max(3.0, vals.std())

    # --- MONTE CARLO CORE ---
    sim_terrains, sim_lengths = [], []
    for i in range(3):
        if i == k_idx:
            sim_terrains.append(np.full(iterations, k_type))
        else:
            p = lap_probs[i] if lap_probs[i] is not None else None
            sim_terrains.append(np.random.choice(TRACK_OPTIONS, size=iterations, p=p))
        
        # AI creates 5,000 different length possibilities for each lap
        sim_lengths.append(np.random.normal(avg_lens[i], std_lens[i], iterations))

    # 100% Length Normalization (The Physics constraint)
    len_matrix = np.column_stack(sim_lengths)
    len_matrix = np.clip(len_matrix, 5, 90)
    len_matrix = (len_matrix.T / len_matrix.sum(axis=1)).T 
    
    terrain_matrix = np.column_stack(sim_terrains)
    results = {}
    for v in vehicles:
        base_speed = np.vectorize(SPEED_DATA[v].get)(terrain_matrix)
        # Apply VPI and random Noise (Mechanical variance)
        final_speed = base_speed * vpi[v] * np.random.normal(1.0, 0.02, (iterations, 3))
        # Total Time = sum of (lap distance / lap speed)
        results[v] = np.sum(len_matrix / final_speed, axis=1)

    # Calculate Probability %
    winners = np.argmin(np.array([results[v] for v in vehicles]), axis=0)
    win_pcts = pd.Series(winners).value_counts(normalize=True).sort_index() * 100
    return {vehicles[i]: win_pcts.get(i, 0) for i in range(3)}, vpi

# --- 4. SIDEBAR (CONTROLS & INPUT) ---
with st.sidebar:
    st.header("🚦 Pre-Race Dashboard")
    
    st.write("### 👁️ Revealed Context")
    pos_map = {"Start (Lap 1)": 0, "Middle (Lap 2)": 1, "Finish (Lap 3)": 2}
    slot_name = st.selectbox("Which slot is shown?", list(pos_map.keys()))
    k_idx, k_type = pos_map[slot_name], st.selectbox("Revealed Track Type", TRACK_OPTIONS)
    
    st.divider()
    st.write("### 🏎️ Vehicle Lineup")
    # Exclusion Selection Logic
    v1_sel = st.selectbox("Vehicle 1", ALL_VEHICLES, index=ALL_VEHICLES.index("Supercar"))
    v2_sel = st.selectbox("Vehicle 2", [v for v in ALL_VEHICLES if v != v1_sel], index=0)
    v3_sel = st.selectbox("Vehicle 3", [v for v in ALL_VEHICLES if v not in [v1_sel, v2_sel]], index=0)
    
    if st.button("🚀 GENERATE PREDICTION", type="primary", use_container_width=True):
        probs, vpi_res = run_master_simulation(v1_sel, v2_sel, v3_sel, k_idx, k_type, history)
        st.session_state['master'] = {'p': probs, 'vpi': vpi_res, 'ctx': {'v': [v1_sel, v2_sel, v3_sel], 'idx': k_idx, 't': k_type}}

    st.divider()
    with st.expander("🛠️ Admin Settings"):
        if st.button("🗑️ Wipe All Memory"):
            if os.path.exists(CSV_FILE): os.remove(CSV_FILE)
            st.rerun()

# --- 5. MAIN DASHBOARD (ACCURACY & METRICS) ---
st.title("🏁 AI RACE PREDICTOR: MASTER EDITION")

# GLOBAL ACCURACY TRACKER & TREND
if not history.empty and 'Predicted' in history.columns and 'Actual' in history.columns:
    valid = history.dropna(subset=['Predicted', 'Actual'])
    valid = valid[~valid['Predicted'].astype(str).isin(['N/A', 'nan'])]
    if not valid.empty:
        acc = (valid['Predicted'].astype(str) == valid['Actual'].astype(str)).mean() * 100
        c1, c2 = st.columns([3, 1])
        with c2: st.metric("🎯 Global Accuracy", f"{acc:.1f}%")
        with c1:
             valid['Correct'] = (valid['Predicted'] == valid['Actual']).astype(int)
             st.write("**Recent Performance Trend (Last 15 Races)**")
             st.line_chart(valid['Correct'].rolling(15).mean() * 100)

if 'master' in st.session_state:
    res = st.session_state['master']
    st.success(f"🧠 **AI Logic:** Anchor Point found at Lap {res['ctx']['idx']+1} ({res['ctx']['t']}). Simulating chain probabilities...")
    
    m_grid = grid(3, vertical_align="center")
    for v, val in res['p'].items():
        vpi_val = (res['vpi'][v] - 1.0) * 100
        label = f"+{vpi_val:.1f}% Boost" if vpi_val > 0 else None
        m_grid.metric(v, f"{val:.1f}%", label)

# --- 6. TELEMETRY LOG (THE TRAINING ROOM) ---
st.divider()
st.subheader("📝 POST-RACE DATA (Critical for AI Evolution)")
ctx = st.session_state.get('master', {'ctx': {'v': [v1_sel, v2_sel, v3_sel], 'idx':0, 't': TRACK_OPTIONS[0]}})['ctx']

with st.form("telemetry_form"):
    st.write("Record actual race data below to teach the AI:")
    winner = st.selectbox("🏆 Actual Winner", ctx['v'])
    
    c_a, c_b, c_c = st.columns(3)
    with c_a:
        s1t = st.selectbox("Lap 1 Track", TRACK_OPTIONS, index=TRACK_OPTIONS.index(ctx['t']) if ctx['idx']==0 else 0)
        s1l = st.number_input("Lap 1 Len %", 1, 100, 33)
    with c_b:
        s2t = st.selectbox("Lap 2 Track", TRACK_OPTIONS, index=TRACK_OPTIONS.index(ctx['t']) if ctx['idx']==1 else 0)
        s2l = st.number_input("Lap 2 Len %", 1, 100, 33)
    with c_c:
        s3t = st.selectbox("Lap 3 Track", TRACK_OPTIONS, index=TRACK_OPTIONS.index(ctx['t']) if ctx['idx']==2 else 0)
        s3l = st.number_input("Lap 3 Len %", 1, 100, 34)

    if st.form_submit_button("💾 SAVE RACE & TRAIN MODEL", use_container_width=True):
        if s1l + s2l + s3l != 100:
            st.error("❌ ERROR: Total length must be exactly 100% to maintain model integrity.")
        else:
            p_val = max(st.session_state['master']['p'], key=st.session_state['master']['p'].get) if 'master' in st.session_state else "N/A"
            new_row = {
                'Vehicle_1': ctx['v'][0], 'Vehicle_2': ctx['v'][1], 'Vehicle_3': ctx['v'][2],
                'Lap_1_Track': s1t, 'Lap_1_Len': s1l, 'Lap_2_Track': s2t, 'Lap_2_Len': s2l,
                'Lap_3_Track': s3t, 'Lap_3_Len': s3l, 'Predicted': p_val, 'Actual': winner
            }
            pd.concat([load_and_migrate_data(), pd.DataFrame([new_row])], ignore_index=True).to_csv(CSV_FILE, index=False)
            st.toast("AI just got smarter!", icon="🧠")
            st.rerun()

# --- 7. DEEP ANALYTICS (VISUALIZING THE BRAIN) ---
if not history.empty:
    st.divider()
    st.header("📊 Deep Learning Analytics")
    tab1, tab2, tab3, tab4 = st.tabs(["🧠 Pattern Brain", "📐 Geometry", "🧬 Vehicle Performance", "📂 Database"])
    
    with tab1:
        st.write("### Sequence Probabilities (What follows what?)")
        if 'Lap_1_Track' in history.columns and 'Lap_2_Track' in history.columns:
            st.write("**Start ➔ Middle Transition**")
            m1 = pd.crosstab(history['Lap_1_Track'], history['Lap_2_Track'], normalize='index') * 100
            st.dataframe(m1.style.format("{:.0f}%").background_gradient(cmap="Blues", axis=1), use_container_width=True)
            
        if 'Lap_2_Track' in history.columns and 'Lap_3_Track' in history.columns:
            st.write("**Middle ➔ Finish Transition**")
            m2 = pd.crosstab(history['Lap_2_Track'], history['Lap_3_Track'], normalize='index') * 100
            st.dataframe(m2.style.format("{:.0f}%").background_gradient(cmap="Greens", axis=1), use_container_width=True)

    with tab2:
        st.write("### Historical Average Length per Track Type")
        # Aggregated stats for geometry learning
        if 'Lap_1_Track' in history.columns:
            geo_stats = history.groupby('Lap_1_Track')[['Lap_1_Len', 'Lap_2_Len', 'Lap_3_Len']].agg(['mean', 'std'])
            st.dataframe(geo_stats.style.format("{:.1f}"))

    with tab3:
        st.write("### Vehicle Performance Index (VPI)")
        if not history.empty and 'Actual' in history.columns and history['Actual'].notna().any():
            win_rec = history['Actual'].value_counts()
            vpi_data = []
            for v in ALL_VEHICLES:
                vpi_data.append({
                    'Vehicle': v, 
                    'Total Wins': win_rec.get(v, 0), 
                    'AI ML Boost': f"x{1.0 + (win_rec.get(v,0)*0.005):.3f}"
                })
            st.dataframe(pd.DataFrame(vpi_data).set_index('Vehicle'), use_container_width=True)
        else:
            st.info("No race winners recorded yet. VPI will update once you save your first race.")

    with tab4:
        st.write("### Bulk Data Merge")
        up = st.file_uploader("Upload CSV to merge into AI brain", type=['csv'])
        if up and st.button("📥 Merge"):
            new_df = pd.read_csv(up)
            # Re-apply migration to new file
            pd.concat([history, new_df], ignore_index=True).to_csv(CSV_FILE, index=False)
            st.rerun()
        st.write("### Raw Database History")
        st.dataframe(history.sort_index(ascending=False), use_container_width=True)
