import streamlit as st
import pandas as pd
import numpy as np
import os
from streamlit_extras.grid import grid

# --- 1. GLOBAL CONFIGURATION (The AI's Physical Knowledge) ---
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

st.set_page_config(layout="wide", page_title="AI Race Predictor Master", page_icon="🏁")

# --- 2. DATA MANAGER (Handles Migration & Cleaning) ---
def load_and_migrate_data():
    if not os.path.exists(CSV_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(CSV_FILE)
        # Remove any junk index columns from previous exports
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # SMART MIGRATION: Convert all previous schema versions to Lap 1/2/3
        rename_map = {
            'V1': 'Vehicle_1', 'V2': 'Vehicle_2', 'V3': 'Vehicle_3',
            'Visible_Track': 'Lap_1_Track', 'Visible_Segment_%': 'Lap_1_Len',
            'Hidden_1_Track': 'Lap_2_Track', 'Hidden_1': 'Lap_2_Track', 'Hidden_1_Len': 'Lap_2_Len',
            'Hidden_2_Track': 'Lap_3_Track', 'Hidden_2': 'Lap_3_Track', 'Hidden_2_Len': 'Lap_3_Len',
            'Stage_1_Track': 'Lap_1_Track', 'Stage_1_Len': 'Lap_1_Len',
            'Stage_2_Track': 'Lap_2_Track', 'Stage_2_Len': 'Lap_2_Len',
            'Stage_3_Track': 'Lap_3_Track', 'Stage_3_Len': 'Lap_3_Len',
            'Predicted_Winner': 'Predicted', 'Actual_Winner': 'Actual'
        }
        df = df.rename(columns=rename_map)
        
        # Ensure numbers are actual numbers
        for col in ['Lap_1_Len', 'Lap_2_Len', 'Lap_3_Len']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception:
        return pd.DataFrame()

# Load DB
history = load_and_migrate_data()

# --- 3. THE "PRE-RACE BLIND" AI ENGINE ---
def run_simulation(v1, v2, v3, known_lap_idx, known_track_type, history_df, iterations=5000):
    vehicles = [v1, v2, v3]
    all_terrains = TRACK_OPTIONS
    
    # 🧠 Learning State
    lap_probs = {0: None, 1: None, 2: None}
    avg_lens = [33.0, 33.0, 34.0]
    std_lens = [8.0, 8.0, 8.0] # High uncertainty for pre-race prediction
    
    if not history_df.empty:
        # ANCHOR LOGIC: Find past races where the user-spotted lap matches
        filter_col = f"Lap_{known_lap_idx + 1}_Track"
        if filter_col in history_df.columns:
            matches = history_df[history_df[filter_col] == known_track_type].tail(50)
            
            if not matches.empty:
                # 1. Learn Track Patterns (What comes before/after the spotted lap?)
                for i in range(3):
                    if i != known_lap_idx:
                        t_col = f"Lap_{i+1}_Track"
                        if t_col in matches.columns:
                            counts = matches[t_col].value_counts(normalize=True)
                            probs = counts.reindex(TRACK_OPTIONS, fill_value=0).values
                            if probs.sum() > 0: lap_probs[i] = probs / probs.sum()
                
                # 2. Learn Length Trends (How long is this visible track usually?)
                for i in range(3):
                    l_col = f"Lap_{i+1}_Len"
                    if l_col in matches.columns:
                        vals = matches[l_col].dropna()
                        if not vals.empty:
                            avg_lens[i] = vals.mean()
                            if len(vals) > 1: std_lens[i] = max(3.0, vals.std())

    # --- MONTE CARLO ---
    stage_terrains = []
    stage_lengths = []
    
    for i in range(3):
        # A. Track Prediction
        if i == known_lap_idx:
            stage_terrains.append(np.full(iterations, known_track_type))
        else:
            p = lap_probs[i] if lap_probs[i] is not None else None
            stage_terrains.append(np.random.choice(all_terrains, size=iterations, p=p))
        
        # B. Length Prediction (Blind estimation based on history)
        stage_lengths.append(np.random.normal(avg_lens[i], std_lens[i], iterations))

    # Normalize Length Matrix to exactly 100%
    len_matrix = np.column_stack(stage_lengths)
    len_matrix = np.clip(len_matrix, 5, 90) # Safety clip
    row_sums = len_matrix.sum(axis=1)
    len_matrix = (len_matrix.T / row_sums).T 
    
    terrain_matrix = np.column_stack(stage_terrains)
    
    # Physics Calculation
    results = {}
    for v in vehicles:
        base_speed = np.vectorize(SPEED_DATA[v].get)(terrain_matrix)
        noise = np.random.normal(1.0, 0.02, (iterations, 3)) # Race variability
        final_speed = base_speed * noise
        # Time = sum of (distance_segment / speed_segment)
        results[v] = np.sum(len_matrix / final_speed, axis=1)

    # Compile Probabilities
    winners = np.argmin(np.array([results[v] for v in vehicles]), axis=0)
    counts = pd.Series(winners).value_counts(normalize=True).sort_index() * 100
    return {vehicles[i]: counts.get(i, 0) for i in range(3)}

# --- 4. SIDEBAR SETUP ---
with st.sidebar:
    st.header("🚦 Pre-Race Setup")
    
    st.write("👁️ **What did you spot?**")
    lap_map = {"Start (Lap 1)": 1, "Middle (Lap 2)": 2, "Finish (Lap 3)": 3}
    c_pos, c_type = st.columns(2)
    with c_pos:
        vis_pos = st.selectbox("Slot", list(lap_map.keys()))
        k_idx = lap_map[vis_pos]
    with c_type:
        k_type = st.selectbox("Track Type", TRACK_OPTIONS)
    
    st.divider()
    # VEHICLE EXCLUSION
    c1 = st.selectbox("Vehicle 1", ALL_VEHICLES, index=ALL_VEHICLES.index("Supercar"))
    v2_list = [v for v in ALL_VEHICLES if v != c1]
    c2 = st.selectbox("Vehicle 2", v2_list, index=0)
    v3_list = [v for v in ALL_VEHICLES if v not in [c1, c2]]
    c3 = st.selectbox("Vehicle 3", v3_list, index=0)
    
    if st.button("🚀 PREDICT WINNER", type="primary", use_container_width=True):
        probs = run_simulation(c1, c2, c3, k_idx-1, k_type, history)
        st.session_state['res'] = {'p': probs, 's': {'v':[c1,c2,c3], 'idx':k_idx, 't':k_type}}

    st.divider()
    # DATA IMPORT
    st.write("📂 **Bulk Import**")
    up = st.file_uploader("Upload CSV", type=['csv'])
    if up and st.button("📥 Merge Files"):
        try:
            new_data = pd.read_csv(up)
            new_data = new_data.loc[:, ~new_data.columns.str.contains('^Unnamed')]
            # Apply standard rename before merging
            final_df = pd.concat([history, new_data], ignore_index=True)
            final_df.to_csv(CSV_FILE, index=False)
            st.success("Data Merged!")
            st.rerun()
        except: st.error("Import failed.")

    if st.button("🗑️ Reset Database"):
        if os.path.exists(CSV_FILE): os.remove(CSV_FILE)
        st.rerun()

# --- 5. MAIN DASHBOARD ---
st.title("🏎️ AI RACE PREDICTOR: MASTER EDITION")

# Live Accuracy Analytics
if not history.empty and 'Predicted' in history.columns and 'Actual' in history.columns:
    valid = history.dropna(subset=['Predicted', 'Actual'])
    valid = valid[~valid['Predicted'].astype(str).isin(['N/A', 'nan'])]
    if not valid.empty:
        acc = (valid['Predicted'].astype(str) == valid['Actual'].astype(str)).mean() * 100
        st.metric("🎯 Prediction Accuracy", f"{acc:.1f}%", help="Based on all recorded history")

if 'res' in st.session_state:
    res = st.session_state['res']
    st.info(f"🧠 **AI Strategy:** Spotting **{res['s']['t']}** at **{vis_pos}**. Thousands of likely track lengths are being calculated.")
    
    m_grid = grid(3, vertical_align="center")
    for v, val in res['p'].items():
        m_grid.metric(v, f"{val:.1f}%")

# --- 6. TELEMETRY LOG (The Training Room) ---
st.divider()
st.subheader("📝 POST-RACE DATA (Critical for Training)")
# Use session state context or defaults
ctx = st.session_state.get('res', {'s': {'v': [c1,c2,c3], 'idx':1, 't':TRACK_OPTIONS[0]}})['s']

with st.form("telemetry_form"):
    st.write("Fill this out AFTER the race starts to teach the AI lengths:")
    winner = st.selectbox("Who actually won?", ctx['v'])
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        s1t = st.selectbox("Lap 1 Track", TRACK_OPTIONS, index=TRACK_OPTIONS.index(ctx['t']) if ctx['idx']==1 else 0)
        s1l = st.number_input("Lap 1 Length %", 1, 100, 33)
    with col_b:
        s2t = st.selectbox("Lap 2 Track", TRACK_OPTIONS, index=TRACK_OPTIONS.index(ctx['t']) if ctx['idx']==2 else 0)
        s2l = st.number_input("Lap 2 Length %", 1, 100, 33)
    with col_c:
        s3t = st.selectbox("Lap 3 Track", TRACK_OPTIONS, index=TRACK_OPTIONS.index(ctx['t']) if ctx['idx']==3 else 0)
        s3l = st.number_input("Lap 3 Length %", 1, 100, 34)

    if st.form_submit_button("💾 SAVE RACE & TRAIN AI", use_container_width=True):
        prediction = max(st.session_state['res']['p'], key=st.session_state['res']['p'].get) if 'res' in st.session_state else "N/A"
        new_row = {
            'Vehicle_1': ctx['v'][0], 'Vehicle_2': ctx['v'][1], 'Vehicle_3': ctx['v'][2],
            'Lap_1_Track': s1t, 'Lap_1_Len': s1l,
            'Lap_2_Track': s2t, 'Lap_2_Len': s2l,
            'Lap_3_Track': s3t, 'Lap_3_Len': s3l,
            'Predicted': prediction, 'Actual': winner
        }
        # Reload, Concat, Save
        updated_db = pd.concat([load_and_migrate_data(), pd.DataFrame([new_row])], ignore_index=True)
        updated_df = updated_db.loc[:, ~updated_db.columns.str.contains('^Unnamed')]
        updated_df.to_csv(CSV_FILE, index=False)
        st.toast("AI just got smarter!", icon="🧠")
        st.rerun()

# --- 7. ANALYTICS (The Visual Proof) ---
if not history.empty:
    st.divider()
    st.subheader("📊 Pattern Recognition Matrix")
    tab1, tab2 = st.tabs(["Transition Matrix", "Raw History"])
    
    with tab1:
        st.write("Probability of Lap 1 ➔ Lap 2 sequence:")
        if 'Lap_1_Track' in history.columns and 'Lap_2_Track' in history.columns:
            m = pd.crosstab(history['Lap_1_Track'], history['Lap_2_Track'], normalize='index') * 100
            st.dataframe(m.style.format("{:.0f}%").background_gradient(cmap="Blues", axis=1), use_container_width=True)
    with tab2:
        st.dataframe(history.sort_index(ascending=False), use_container_width=True)
