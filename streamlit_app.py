import streamlit as st
import pandas as pd
import numpy as np
import os
from streamlit_extras.grid import grid

# --- 1. PHYSICAL ENGINE CONFIGURATION ---
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

st.set_page_config(layout="wide", page_title="AI Race Predictor Master", page_icon="🏎️")

# --- 2. DATA ARCHITECT (FIXED KEYERROR) ---
def load_and_migrate_data():
    if not os.path.exists(CSV_FILE): return pd.DataFrame()
    try:
        df = pd.read_csv(CSV_FILE)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Aggressive Mapper for all possible variations found in your export
        rename_map = {
            'V1': 'Vehicle_1', 'V2': 'Vehicle_2', 'V3': 'Vehicle_3',
            'Visible_Track': 'Lap_1_Track', 'Visible_Segment_%': 'Lap_1_Len',
            'Hidden_1_Track': 'Lap_2_Track', 'Hidden_1_Len': 'Lap_2_Len',
            'Hidden_2_Track': 'Lap_3_Track', 'Hidden_2_Len': 'Lap_3_Len',
            'Stage_1_Track': 'Lap_1_Track', 'Stage_2_Track': 'Lap_2_Track', 'Stage_3_Track': 'Lap_3_Track',
            'Stage_1_Len': 'Lap_1_Len', 'Stage_2_Len': 'Lap_2_Len', 'Stage_3_Len': 'Lap_3_Len'
        }
        df = df.rename(columns=rename_map)
        
        # Critical Fix: Ensure Lap_X_Len exists for the simulation engine
        for i in range(1, 4):
            if f'Lap_{i}_Len' not in df.columns:
                df[f'Lap_{i}_Len'] = 33.3 # Default if column is missing
            df[f'Lap_{i}_Len'] = pd.to_numeric(df[f'Lap_{i}_Len'], errors='coerce').fillna(33.3)
            
        return df
    except Exception: return pd.DataFrame()

history = load_and_migrate_data()

# --- 3. THE BIDIRECTIONAL AI ENGINE ---
def run_master_simulation(v1, v2, v3, k_idx, k_type, history_df, iterations=5000):
    vehicles = [v1, v2, v3]
    lap_probs = {0: None, 1: None, 2: None}
    avg_lens = [33.3, 33.3, 33.4]
    std_lens = [8.0, 8.0, 8.0]
    
    if not history_df.empty:
        filter_col = f"Lap_{k_idx + 1}_Track"
        if filter_col in history_df.columns:
            matches = history_df[history_df[filter_col] == k_type].tail(50)
            
            if not matches.empty:
                for i in range(3):
                    # Learn Tracks
                    if i != k_idx:
                        t_col = f"Lap_{i+1}_Track"
                        if t_col in matches.columns:
                            counts = matches[t_col].value_counts(normalize=True)
                            probs = counts.reindex(TRACK_OPTIONS, fill_value=0).values
                            if probs.sum() > 0: lap_probs[i] = probs / probs.sum()
                    
                    # Learn Lengths (Error was here - now hardened)
                    l_col = f"Lap_{i+1}_Len"
                    if l_col in matches.columns:
                        vals = matches[l_col].dropna()
                        if not vals.empty:
                            avg_lens[i] = vals.mean()
                            if len(vals) > 1: std_lens[i] = max(3.0, vals.std())

    sim_terrains = []
    sim_lengths = []
    
    for i in range(3):
        if i == k_idx:
            sim_terrains.append(np.full(iterations, k_type))
        else:
            p = lap_probs[i] if lap_probs[i] is not None else None
            sim_terrains.append(np.random.choice(TRACK_OPTIONS, size=iterations, p=p))
        
        sim_lengths.append(np.random.normal(avg_lens[i], std_lens[i], iterations))

    len_matrix = np.column_stack(sim_lengths)
    len_matrix = np.clip(len_matrix, 5, 90)
    len_matrix = (len_matrix.T / len_matrix.sum(axis=1)).T 
    
    terrain_matrix = np.column_stack(sim_terrains)
    
    results = {}
    for v in vehicles:
        base_speed = np.vectorize(SPEED_DATA[v].get)(terrain_matrix)
        noise = np.random.normal(1.0, 0.02, (iterations, 3))
        final_speed = base_speed * noise
        results[v] = np.sum(len_matrix / final_speed, axis=1)

    winners = np.argmin(np.array([results[v] for v in vehicles]), axis=0)
    win_pcts = pd.Series(winners).value_counts(normalize=True).sort_index() * 100
    return {vehicles[i]: win_pcts.get(i, 0) for i in range(3)}

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("🚦 Race Context")
    lap_options = {"Start (Lap 1)": 0, "Middle (Lap 2)": 1, "Finish (Lap 3)": 2}
    slot_name = st.selectbox("Which lap is revealed?", list(lap_options.keys()))
    k_idx = lap_options[slot_name]
    k_type = st.selectbox("Revealed Track Type", TRACK_OPTIONS)
    
    st.divider()
    v1_sel = st.selectbox("Vehicle 1", ALL_VEHICLES, index=ALL_VEHICLES.index("Supercar"))
    v2_opts = [v for v in ALL_VEHICLES if v != v1_sel]
    v2_sel = st.selectbox("Vehicle 2", v2_opts, index=0)
    v3_opts = [v for v in ALL_VEHICLES if v not in [v1_sel, v2_sel]]
    v3_sel = st.selectbox("Vehicle 3", v3_opts, index=0)
    
    if st.button("🚀 RUN PREDICTION", type="primary", use_container_width=True):
        probs = run_master_simulation(v1_sel, v2_sel, v3_sel, k_idx, k_type, history)
        st.session_state['master_res'] = {
            'probs': probs, 
            'ctx': {'v': [v1_sel, v2_sel, v3_sel], 'idx': k_idx, 't': k_type}
        }

    st.divider()
    if st.button("🗑️ Reset All Training Data"):
        if os.path.exists(CSV_FILE): os.remove(CSV_FILE)
        st.rerun()

# --- 5. MAIN DASHBOARD ---
st.title("🏁 AI RACE PREDICTOR: MASTER EDITION")

if not history.empty and 'Predicted' in history.columns and 'Actual' in history.columns:
    valid = history.dropna(subset=['Predicted', 'Actual'])
    valid = valid[~valid['Predicted'].astype(str).isin(['N/A', 'nan'])]
    if not valid.empty:
        acc = (valid['Predicted'].astype(str) == valid['Actual'].astype(str)).mean() * 100
        st.metric("🎯 Global Prediction Accuracy", f"{acc:.1f}%")

if 'master_res' in st.session_state:
    res = st.session_state['master_res']
    st.info(f"🧠 **AI Context:** Lap {res['ctx']['idx']+1} is {res['ctx']['t']}.")
    m_grid = grid(3, vertical_align="center")
    for v, val in res['probs'].items():
        m_grid.metric(v, f"{val:.1f}%")

# --- 6. POST-RACE TELEMETRY ---
st.divider()
st.subheader("📝 POST-RACE REPORT (Record Actuals)")
ctx = st.session_state.get('master_res', {'ctx': {'v': [v1_sel, v2_sel, v3_sel], 'idx':0, 't': TRACK_OPTIONS[0]}})['ctx']

with st.form("telemetry_form"):
    winner = st.selectbox("🏆 Winner", ctx['v'])
    c_a, c_b, c_c = st.columns(3)
    with c_a:
        s1t = st.selectbox("Lap 1 Track", TRACK_OPTIONS, index=TRACK_OPTIONS.index(ctx['t']) if ctx['idx']==0 else 0)
        s1l = st.number_input("Lap 1 Length %", 1, 100, 33)
    with c_b:
        s2t = st.selectbox("Lap 2 Track", TRACK_OPTIONS, index=TRACK_OPTIONS.index(ctx['t']) if ctx['idx']==1 else 0)
        s2l = st.number_input("Lap 2 Length %", 1, 100, 33)
    with c_c:
        s3t = st.selectbox("Lap 3 Track", TRACK_OPTIONS, index=TRACK_OPTIONS.index(ctx['t']) if ctx['idx']==2 else 0)
        s3l = st.number_input("Lap 3 Length %", 1, 100, 34)

    if st.form_submit_button("💾 SAVE RACE & TRAIN AI", use_container_width=True):
        pred = max(st.session_state['master_res']['probs'], key=st.session_state['master_res']['probs'].get) if 'master_res' in st.session_state else "N/A"
        new_row = {
            'Vehicle_1': ctx['v'][0], 'Vehicle_2': ctx['v'][1], 'Vehicle_3': ctx['v'][2],
            'Lap_1_Track': s1t, 'Lap_1_Len': s1l, 'Lap_2_Track': s2t, 'Lap_2_Len': s2l, 'Lap_3_Track': s3t, 'Lap_3_Len': s3l,
            'Predicted': pred, 'Actual': winner
        }
        updated_df = pd.concat([load_and_migrate_data(), pd.DataFrame([new_row])], ignore_index=True)
        updated_df.to_csv(CSV_FILE, index=False)
        st.toast("AI Model Updated!", icon="🧠")
        st.rerun()

# --- 7. DEEP ANALYTICS ---
if not history.empty:
    st.divider()
    st.header("📊 AI Chain Analytics")
    tab1, tab2, tab3 = st.tabs(["Sequence Matrix", "Import Tool", "Raw Database"])
    with tab1:
        if 'Lap_1_Track' in history.columns and 'Lap_2_Track' in history.columns:
            m1 = pd.crosstab(history['Lap_1_Track'], history['Lap_2_Track'], normalize='index') * 100
            st.dataframe(m1.style.format("{:.0f}%").background_gradient(cmap="Blues", axis=1), use_container_width=True)
    with tab2:
        up = st.file_uploader("Merge External CSV", type=['csv'])
        if up and st.button("📥 Process & Merge"):
            new_csv = pd.read_csv(up)
            # Standardize before merge
            new_csv = new_csv.rename(columns={'V1': 'Vehicle_1', 'V2': 'Vehicle_2', 'V3': 'Vehicle_3', 'Visible_Segment_%': 'Lap_1_Len', 'Hidden_1_Len': 'Lap_2_Len', 'Hidden_2_Len': 'Lap_3_Len'})
            merged = pd.concat([history, new_csv], ignore_index=True)
            merged.to_csv(CSV_FILE, index=False)
            st.rerun()
    with tab3:
        st.dataframe(history.sort_index(ascending=False), use_container_width=True)
