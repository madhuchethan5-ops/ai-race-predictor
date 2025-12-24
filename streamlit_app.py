import streamlit as st
import pandas as pd
import numpy as np
import os
from streamlit_extras.grid import grid

# --- 1. GLOBAL CONFIGURATION ---
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

st.set_page_config(layout="wide", page_title="AI Race Predictor: 3-Lap Master", page_icon="🏁")

# --- 2. INTELLIGENT DATA MANAGER ---
def load_and_migrate_data():
    if not os.path.exists(CSV_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(CSV_FILE)
        
        # A. Clean Unnamed
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # B. Detect & Migrate Schema
        # Convert all previous formats to "Lap 1/2/3"
        rename_map = {
            'Predicted_Winner': 'Predicted', 'Actual_Winner': 'Actual', 
            'V1': 'Vehicle_1', 'V2': 'Vehicle_2', 'V3': 'Vehicle_3',
            'Visible_Track': 'Lap_1_Track', 'Visible_Segment_%': 'Lap_1_Len',
            'Hidden_1_Track': 'Lap_2_Track', 'Hidden_1': 'Lap_2_Track', 'Hidden_1_Len': 'Lap_2_Len',
            'Hidden_2_Track': 'Lap_3_Track', 'Hidden_2': 'Lap_3_Track', 'Hidden_2_Len': 'Lap_3_Len',
            'Stage_1_Track': 'Lap_1_Track', 'Stage_1_Len': 'Lap_1_Len',
            'Stage_2_Track': 'Lap_2_Track', 'Stage_2_Len': 'Lap_2_Len',
            'Stage_3_Track': 'Lap_3_Track', 'Stage_3_Len': 'Lap_3_Len'
        }
        df = df.rename(columns=rename_map)
        
        # C. Ensure Columns
        required_cols = [
            'Lap_1_Track', 'Lap_1_Len',
            'Lap_2_Track', 'Lap_2_Len',
            'Lap_3_Track', 'Lap_3_Len',
            'Predicted', 'Actual', 'Vehicle_1', 'Vehicle_2', 'Vehicle_3'
        ]
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan
                
        # D. Type Casting
        num_cols = ['Lap_1_Len', 'Lap_2_Len', 'Lap_3_Len']
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df
    except Exception as e:
        return pd.DataFrame()

history = load_and_migrate_data()

# --- 3. PATTERN RECOGNITION AI ---
def run_simulation(v1, v2, v3, known_lap_idx, known_track_type, history_df, iterations=5000):
    vehicles = [v1, v2, v3]
    all_terrains = TRACK_OPTIONS
    
    # 1. Pattern Learning (Markov Chain)
    lap_probs = {0: None, 1: None, 2: None}
    avg_lens = [0.33, 0.33, 0.34]
    std_lens = [0.05, 0.05, 0.05]
    
    if not history_df.empty:
        filter_col = f"Lap_{known_lap_idx + 1}_Track"
        if filter_col in history_df.columns:
            matches = history_df[history_df[filter_col] == known_track_type].tail(50)
            
            if not matches.empty:
                for i in range(3):
                    if i == known_lap_idx: continue
                    target_col = f"Lap_{i+1}_Track"
                    if target_col in matches.columns:
                        counts = matches[target_col].value_counts(normalize=True)
                        probs = counts.reindex(TRACK_OPTIONS, fill_value=0).values
                        if probs.sum() > 0: lap_probs[i] = probs / probs.sum()
                
                for i in range(3):
                    len_col = f"Lap_{i+1}_Len"
                    if len_col in matches.columns:
                        vals = matches[len_col].dropna()
                        if not vals.empty:
                            avg_lens[i] = vals.mean() / 100.0
                            if len(vals) > 1: std_lens[i] = max(0.02, vals.std() / 100.0)

    # 2. Monte Carlo
    lap_terrains = []
    lap_lengths = []
    
    for i in range(3):
        if i == known_lap_idx:
            lap_terrains.append(np.full(iterations, known_track_type))
        else:
            if lap_probs[i] is not None:
                lap_terrains.append(np.random.choice(all_terrains, size=iterations, p=lap_probs[i]))
            else:
                lap_terrains.append(np.random.choice(all_terrains, size=iterations))
        
        raw_len = np.random.normal(avg_lens[i], std_lens[i], iterations)
        lap_lengths.append(np.clip(raw_len, 0.1, 0.8))

    total_len = lap_lengths[0] + lap_lengths[1] + lap_lengths[2]
    lap_lengths[0] /= total_len
    lap_lengths[1] /= total_len
    lap_lengths[2] /= total_len
    
    terrain_matrix = np.column_stack(lap_terrains)
    
    results = {}
    for v in vehicles:
        base_speed = np.vectorize(SPEED_DATA[v].get)(terrain_matrix)
        noise = np.random.normal(1.0, 0.02, (iterations, 3))
        final_speed = base_speed * noise
        
        times = (lap_lengths[0]/final_speed[:, 0]) + \
                (lap_lengths[1]/final_speed[:, 1]) + \
                (lap_lengths[2]/final_speed[:, 2])
        
        results[v] = times

    winners = np.argmin(np.array([results[v] for v in vehicles]), axis=0)
    counts = pd.Series(winners).value_counts(normalize=True).sort_index() * 100
    
    return {vehicles[i]: counts.get(i, 0) for i in range(3)}

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("🚦 3-Lap Race Setup")
    
    st.write("👁️ **What do you see?**")
    c_vis, c_type = st.columns([1, 1.5])
    with c_vis:
        known_lap = st.selectbox("Visible Lap", [1, 2, 3], help="Which lap is currently visible?")
    with c_type:
        known_type = st.selectbox("Track Type", TRACK_OPTIONS)
        
    st.divider()
    
    # VEHICLE SELECTION (Exclusion Logic)
    c1 = st.selectbox("Vehicle 1", ALL_VEHICLES, index=ALL_VEHICLES.index("Supercar"))
    v2_list = [v for v in ALL_VEHICLES if v != c1]
    c2 = st.selectbox("Vehicle 2", v2_list, index=0)
    v3_list = [v for v in ALL_VEHICLES if v not in [c1, c2]]
    c3 = st.selectbox("Vehicle 3", v3_list, index=0)
    
    if st.button("🚀 PREDICT RESULT", type="primary", use_container_width=True):
        probs = run_simulation(c1, c2, c3, known_lap-1, known_type, history)
        st.session_state['last_probs'] = probs
        st.session_state['last_setup'] = {'v': [c1, c2, c3], 'k_lap': known_lap, 'k_type': known_type}
        st.session_state['has_predicted'] = True

    st.divider()
    
    # IMPORT
    st.write("📂 **Import Data**")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded_file is not None and st.button("📥 Import"):
        try:
            new_data = pd.read_csv(uploaded_file)
            new_data = new_data.loc[:, ~new_data.columns.str.contains('^Unnamed')]
            
            rename_map = {
                'Predicted_Winner': 'Predicted', 'Actual_Winner': 'Actual', 
                'V1': 'Vehicle_1', 'V2': 'Vehicle_2', 'V3': 'Vehicle_3',
                'Visible_Track': 'Lap_1_Track', 'Visible_Segment_%': 'Lap_1_Len',
                'Hidden_1_Track': 'Lap_2_Track', 'Hidden_1': 'Lap_2_Track', 'Hidden_1_Len': 'Lap_2_Len',
                'Hidden_2_Track': 'Lap_3_Track', 'Hidden_2': 'Lap_3_Track', 'Hidden_2_Len': 'Lap_3_Len',
                'Stage_1_Track': 'Lap_1_Track', 'Stage_1_Len': 'Lap_1_Len',
                'Stage_2_Track': 'Lap_2_Track', 'Stage_2_Len': 'Lap_2_Len',
                'Stage_3_Track': 'Lap_3_Track', 'Stage_3_Len': 'Lap_3_Len'
            }
            new_data = new_data.rename(columns=rename_map)
            
            if os.path.exists(CSV_FILE):
                master = pd.read_csv(CSV_FILE)
                final = pd.concat([master, new_data], ignore_index=True)
            else:
                final = new_data
                
            final.to_csv(CSV_FILE, index=False)
            st.success("Imported!")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")
            
    if st.button("🗑️ Reset All Data"):
        if os.path.exists(CSV_FILE): os.remove(CSV_FILE)
        st.rerun()

# --- 5. DASHBOARD ---
st.title("🏁 AI RACE PREDICTOR")

# Calculate & Display Accuracy at the Top
if not history.empty and 'Predicted' in history.columns and 'Actual' in history.columns:
    valid = history.dropna(subset=['Predicted', 'Actual'])
    # Clean string data
    valid['Predicted'] = valid['Predicted'].astype(str)
    valid['Actual'] = valid['Actual'].astype(str)
    # Filter out N/A
    valid = valid[valid['Predicted'] != 'N/A']
    valid = valid[valid['Predicted'] != 'nan']
    
    if not valid.empty:
        correct = (valid['Predicted'] == valid['Actual']).sum()
        total = len(valid)
        acc = (correct / total) * 100
        
        col1, col2 = st.columns([3, 1])
        with col1:
             st.caption(f"Based on {total} recorded races")
        with col2:
             st.metric("🎯 AI Accuracy", f"{acc:.1f}%")

if st.session_state.get('has_predicted', False):
    probs = st.session_state['last_probs']
    setup = st.session_state['last_setup']
    
    st.info(f"🧠 **AI Analysis:** Knowing **Lap {setup['k_lap']}** is **{setup['k_type']}**, simulated 5,000 scenarios.")
    
    m_grid = grid(3, vertical_align="center")
    for veh, val in probs.items():
        m_grid.metric(veh, f"{val:.1f}%")
        
    gap = max(probs.values()) - sorted(probs.values())[-2]
    if gap > 35: st.success("Confident Win")
    else: st.warning("Close Race")

# --- 6. TELEMETRY LOGGING ---
st.divider()
st.subheader("📝 RECORD RACE (Feeds the AI)")
setup = st.session_state.get('last_setup', {'v': [c1, c2, c3], 'k_lap':1, 'k_type': 'Dirt'})

with st.form("telemetry"):
    st.write("Who Won?")
    winner = st.selectbox("Winner", setup['v'])
    
    st.write("---")
    # LAP 1
    c1a, c1b = st.columns(2)
    with c1a: s1_t = st.selectbox("Lap 1 Track", TRACK_OPTIONS, index=TRACK_OPTIONS.index(setup['k_type']) if setup['k_lap']==1 else 0)
    with c1b: s1_l = st.number_input("Lap 1 Length %", 0, 100, 33)
    
    # LAP 2
    c2a, c2b = st.columns(2)
    with c2a: s2_t = st.selectbox("Lap 2 Track", TRACK_OPTIONS, index=TRACK_OPTIONS.index(setup['k_type']) if setup['k_lap']==2 else 0)
    with c2b: s2_l = st.number_input("Lap 2 Length %", 0, 100, 33)

    # LAP 3
    c3a, c3b = st.columns(2)
    with c3a: s3_t = st.selectbox("Lap 3 Track", TRACK_OPTIONS, index=TRACK_OPTIONS.index(setup['k_type']) if setup['k_lap']==3 else 0)
    with c3b: s3_l = st.number_input("Lap 3 Length %", 0, 100, 34)

    if st.form_submit_button("💾 SAVE & TRAIN AI", use_container_width=True):
        row = {
            'Vehicle_1': str(setup['v'][0]), 'Vehicle_2': str(setup['v'][1]), 'Vehicle_3': str(setup['v'][2]),
            'Lap_1_Track': s1_t, 'Lap_1_Len': s1_l,
            'Lap_2_Track': s2_t, 'Lap_2_Len': s2_l,
            'Lap_3_Track': s3_t, 'Lap_3_Len': s3_l,
            'Predicted': max(st.session_state.get('last_probs', {}), key=st.session_state.get('last_probs', {}).get, default="N/A"),
            'Actual': str(winner)
        }
        
        try:
            curr = load_and_migrate_data()
            new = pd.DataFrame([row])
            final = pd.concat([curr, new], ignore_index=True)
            final.to_csv(CSV_FILE, index=False)
            st.toast("Model Updated!", icon="🧠")
            st.rerun()
        except Exception as e:
            st.error(f"Save Failed: {e}")

# --- 7. ANALYTICS ---
if not history.empty:
    st.divider()
    st.header("📊 Deep Learning Analytics")
    
    t1, t2 = st.tabs(["🧠 Pattern Matrix", "📂 Raw Data"])
    
    with t1:
        st.write("### Lap Transition Probability")
        st.write("If **Lap 1** is Row, how likely is **Lap 2** (Column)?")
        if 'Lap_1_Track' in history.columns and 'Lap_2_Track' in history.columns:
            matrix = pd.crosstab(history['Lap_1_Track'], history['Lap_2_Track'], normalize='index') * 100
            st.dataframe(matrix.style.background_gradient(cmap="Greens", axis=1).format("{:.0f}%"), use_container_width=True)
            
    with t2:
        st.dataframe(history.sort_index(ascending=False), use_container_width=True)
