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

# SORTED MASTER LISTS
ALL_VEHICLES = sorted(list(SPEED_DATA.keys()))
TRACK_OPTIONS = sorted(list(SPEED_DATA["Car"].keys()))
VALID_TRACKS = TRACK_OPTIONS

CSV_FILE = 'race_history.csv'
st.set_page_config(layout="wide", page_title="AI Race Predictor Pro", page_icon="🏎️")

# --- 2. DATA MANAGER ---
def load_clean_history():
    if not os.path.exists(CSV_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(CSV_FILE)
        
        # Standardize Columns
        rename_map = {
            'Predicted_Winner': 'Predicted', 'Actual_Winner': 'Actual', 
            'Visible_%': 'Visible_Segment_%', 'Visible_Lane_Length (%)': 'Visible_Segment_%',
            'Hidden_1': 'Hidden_1_Track', 'Hidden_2': 'Hidden_2_Track',
            'V1': 'Vehicle_1', 'V2': 'Vehicle_2', 'V3': 'Vehicle_3', 'Lane': 'Active_Lane'
        }
        df = df.rename(columns=rename_map)
        
        # Merge Hidden Track Logic
        if 'Hidden_1_Track' not in df.columns: df['Hidden_1_Track'] = np.nan
        if 'Hidden_1' in df.columns: 
            df['Hidden_1_Track'] = df['Hidden_1_Track'].fillna(df['Hidden_1'])
            df = df.drop(columns=['Hidden_1'])

        if 'Hidden_2_Track' not in df.columns: df['Hidden_2_Track'] = np.nan
        if 'Hidden_2' in df.columns: 
            df['Hidden_2_Track'] = df['Hidden_2_Track'].fillna(df['Hidden_2'])
            df = df.drop(columns=['Hidden_2'])
        
        # Validate
        if 'Visible_Track' in df.columns:
            df = df[df['Visible_Track'].isin(VALID_TRACKS)]
        if 'Visible_Segment_%' in df.columns:
            df['Visible_Segment_%'] = pd.to_numeric(df['Visible_Segment_%'], errors='coerce')
            df = df.dropna(subset=['Visible_Segment_%'])
            
        return df
    except Exception:
        return pd.DataFrame()

history = load_clean_history()

# --- 3. PATTERN RECOGNITION ENGINE ---
def run_simulation(v1, v2, v3, visible_t, visible_l, history_df, iterations=5000):
    vehicles = [v1, v2, v3]
    all_terrains = TRACK_OPTIONS
    
    # Defaults
    avg_vis = 0.33
    vis_std = 0.12
    h1_probs = None 
    h2_probs = None
    avg_h1_split = 0.5 
    std_h1_split = 0.15

    # LEARN FROM HISTORY
    if not history_df.empty:
        match = history_df[history_df['Visible_Track'] == visible_t].tail(50)
        
        if not match.empty:
            # A. VISIBLE LENGTH
            clean_vis = pd.to_numeric(match['Visible_Segment_%'], errors='coerce').dropna()
            if not clean_vis.empty:
                 avg_vis = clean_vis.mean() / 100
                 if len(clean_vis) > 1: vis_std = max(0.04, clean_vis.std() / 100)

            # B. HIDDEN PATTERNS
            if 'Hidden_1_Track' in match.columns:
                h1_counts = match['Hidden_1_Track'].value_counts(normalize=True)
                h1_probs = h1_counts.reindex(TRACK_OPTIONS, fill_value=0).values
                if h1_probs.sum() > 0: h1_probs = h1_probs / h1_probs.sum()
                else: h1_probs = None 

            if 'Hidden_2_Track' in match.columns:
                h2_counts = match['Hidden_2_Track'].value_counts(normalize=True)
                h2_probs = h2_counts.reindex(TRACK_OPTIONS, fill_value=0).values
                if h2_probs.sum() > 0: h2_probs = h2_probs / h2_probs.sum()
                else: h2_probs = None

            # C. HIDDEN SPLIT
            if 'Hidden_1_Len' in match.columns and 'Hidden_2_Len' in match.columns:
                h1_len = pd.to_numeric(match['Hidden_1_Len'], errors='coerce').fillna(33)
                h2_len = pd.to_numeric(match['Hidden_2_Len'], errors='coerce').fillna(33)
                total_hidden = h1_len + h2_len
                valid_splits = total_hidden > 0
                if valid_splits.any():
                    ratios = h1_len[valid_splits] / total_hidden[valid_splits]
                    avg_h1_split = ratios.mean()
                    if len(ratios) > 1: std_h1_split = max(0.05, ratios.std())

    # MONTE CARLO
    vis_lens = np.clip(np.random.normal(avg_vis, vis_std, iterations), 0.05, 0.95)
    remaining = 1.0 - vis_lens

    h1_ratios = np.clip(np.random.normal(avg_h1_split, std_h1_split, iterations), 0.1, 0.9)
    h1_lens = remaining * h1_ratios
    h2_lens = remaining - h1_lens

    if h1_probs is not None: h1_terrains = np.random.choice(all_terrains, size=iterations, p=h1_probs)
    else: h1_terrains = np.random.choice(all_terrains, size=iterations)

    if h2_probs is not None: h2_terrains = np.random.choice(all_terrains, size=iterations, p=h2_probs)
    else: h2_terrains = np.random.choice(all_terrains, size=iterations)
        
    seg_terrains = np.column_stack([np.full(iterations, visible_t), h1_terrains, h2_terrains])

    results = {}
    for v in vehicles:
        speed_lookup = np.vectorize(SPEED_DATA[v].get)(seg_terrains)
        noise = np.random.normal(1.0, 0.02, (iterations, 3))
        noisy_speeds = speed_lookup * noise
        times = (vis_lens/noisy_speeds[:, 0]) + (h1_lens/noisy_speeds[:, 1]) + (h2_lens/noisy_speeds[:, 2])
        results[v] = times

    winners = np.argmin(np.array([results[v] for v in vehicles]), axis=0)
    counts = pd.Series(winners).value_counts(normalize=True).sort_index() * 100
    
    return {vehicles[i]: counts.get(i, 0) for i in range(3)}

# --- 4. SIDEBAR (DYNAMIC SELECTION) ---
with st.sidebar:
    st.header("🚦 Race Setup")
    v_track = st.selectbox("Visible Track", TRACK_OPTIONS)
    v_lane = st.radio("Active Lane", [1, 2, 3], horizontal=True)
    st.divider()
    
    # --- DYNAMIC VEHICLE SELECTORS ---
    # Vehicle 1
    c1 = st.selectbox("Vehicle 1 (Top)", ALL_VEHICLES, index=ALL_VEHICLES.index("Supercar"))
    
    # Vehicle 2 (Remove C1 from options)
    v2_options = [v for v in ALL_VEHICLES if v != c1]
    # Ensure default index is valid
    default_v2 = "Sports Car" if "Sports Car" in v2_options else v2_options[0]
    c2 = st.selectbox("Vehicle 2 (Mid)", v2_options, index=v2_options.index(default_v2))
    
    # Vehicle 3 (Remove C1 and C2 from options)
    v3_options = [v for v in ALL_VEHICLES if v not in [c1, c2]]
    default_v3 = "Car" if "Car" in v3_options else v3_options[0]
    c3 = st.selectbox("Vehicle 3 (Bot)", v3_options, index=v3_options.index(default_v3))
    
    # Button Logic using Session State
    if st.button("🚀 PREDICT OUTCOME", type="primary", use_container_width=True):
        probs = run_simulation(c1, c2, c3, v_track, v_lane, history)
        st.session_state['last_probs'] = probs
        st.session_state['last_vehicles'] = [c1, c2, c3]
        st.session_state['last_lane'] = v_lane
        st.session_state['has_predicted'] = True
    
    st.divider()
    
    # --- IMPORT TOOL (UPDATED) ---
    st.write("📂 **Import & Fix Data**")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded_file is not None:
        if st.button("📥 Process & Merge"):
            try:
                # Read CSV, usually defaults to comma separator
                new_data = pd.read_csv(uploaded_file)
                
                # Check if it's the new format (which has correct names) or old
                # Use standard renaming just in case to be safe
                rename_map = {
                    'Predicted_Winner': 'Predicted', 'Actual_Winner': 'Actual', 
                    'Visible_%': 'Visible_Segment_%', 'Visible_Lane_Length (%)': 'Visible_Segment_%',
                    'Hidden_1': 'Hidden_1_Track', 'Hidden_2': 'Hidden_2_Track',
                    'V1': 'Vehicle_1', 'V2': 'Vehicle_2', 'V3': 'Vehicle_3', 'Lane': 'Active_Lane'
                }
                new_data = new_data.rename(columns=rename_map)
                
                # Remove unnamed columns (like index columns from export)
                new_data = new_data.loc[:, ~new_data.columns.str.contains('^Unnamed')]

                if os.path.exists(CSV_FILE):
                    master_df = pd.read_csv(CSV_FILE)
                else:
                    master_df = pd.DataFrame()
                
                combined_df = pd.concat([master_df, new_data], ignore_index=True)
                
                # Run the merge logic again on the whole set
                if 'Hidden_1_Track' not in combined_df.columns: combined_df['Hidden_1_Track'] = np.nan
                if 'Hidden_1' in combined_df.columns: 
                    combined_df['Hidden_1_Track'] = combined_df['Hidden_1_Track'].fillna(combined_df['Hidden_1'])
                if 'Hidden_2_Track' not in combined_df.columns: combined_df['Hidden_2_Track'] = np.nan
                if 'Hidden_2' in combined_df.columns: 
                    combined_df['Hidden_2_Track'] = combined_df['Hidden_2_Track'].fillna(combined_df['Hidden_2'])
                
                combined_df.to_csv(CSV_FILE, index=False)
                st.success(f"Merged! Total Records: {len(combined_df)}")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    with st.expander("🛠️ Admin Tools"):
        if st.button("🗑️ Force Wipe Database"):
            if os.path.exists(CSV_FILE):
                os.remove(CSV_FILE)
                st.rerun()

# --- 5. DASHBOARD ---
st.title("🏎️ AI RACE PREDICTOR PRO")

# Display Results if a prediction exists in session state
if st.session_state.get('has_predicted', False):
    probs = st.session_state['last_probs']
    
    m_grid = grid(3, vertical_align="center")
    for veh, val in probs.items():
        m_grid.metric(veh, f"{val:.1f}%")

    gap = max(probs.values()) - sorted(probs.values())[-2]
    if gap > 35: st.success("🏁 HIGH CONFIDENCE PREDICTION")
    elif gap > 15: st.warning("⚠️ MODERATE RISK")
    else: st.error("⚡ EXTREME VOLATILITY")

# --- 6. TELEMETRY LOGGING (SAVES FULL CONTEXT) ---
st.divider()
st.subheader("📝 POST-RACE TELEMETRY")
logger_vehicles = st.session_state.get('last_vehicles', [c1, c2, c3])

with st.form("logger_form", clear_on_submit=False):
    c_a, c_b = st.columns(2)
    with c_a: winner = st.selectbox("Actual Winner", logger_vehicles)
    with c_b: v_len = st.number_input("Visible Segment Length %", 0.0, 100.0, 33.0, step=1.0)
    
    c_c, c_d = st.columns(2)
    with c_c: h1_t = st.selectbox("Hidden 1 Type", TRACK_OPTIONS)
    with c_d: h1_l = st.number_input("Hidden 1 Length %", 0.0, 100.0, 33.0, step=1.0)
    
    c_e, c_f = st.columns(2)
    with c_e: h2_t = st.selectbox("Hidden 2 Type", TRACK_OPTIONS)
    with c_f: h2_l = st.number_input("Hidden 2 Length %", 0.0, 100.0, 34.0, step=1.0)

    # RENAMED BUTTON AS REQUESTED
    submitted = st.form_submit_button("💾 SAVE RACE RESULT", use_container_width=True)
    
    if submitted:
        last_probs = st.session_state.get('last_probs', {})
        predicted = max(last_probs, key=last_probs.get) if last_probs else "N/A"
        
        saved_v1, saved_v2, saved_v3 = st.session_state.get('last_vehicles', [c1, c2, c3])
        saved_lane = st.session_state.get('last_lane', v_lane)

        log_entry = {
            "Vehicle_1": str(saved_v1), "Vehicle_2": str(saved_v2), "Vehicle_3": str(saved_v3),
            "Active_Lane": int(saved_lane),
            "Visible_Track": str(v_track), "Visible_Segment_%": float(v_len),
            "Hidden_1_Track": str(h1_t), "Hidden_1_Len": float(h1_l),
            "Hidden_2_Track": str(h2_t), "Hidden_2_Len": float(h2_l),
            "Predicted": str(predicted), "Actual": str(winner)
        }
        
        if log_entry["Visible_Track"] in VALID_TRACKS:
            try:
                current_df = load_clean_history() 
                new_row = pd.DataFrame([log_entry])
                updated_df = pd.concat([current_df, new_row], ignore_index=True)
                updated_df.to_csv(CSV_FILE, index=False)
                st.toast(f"Saved Full Race Context! Total: {len(updated_df)}", icon="✅")
                # We do NOT rerun here immediately so you can see the toast, 
                # but you might want to manually refresh or click predict again.
            except Exception as e:
                st.error(f"Save Error: {e}")
        else:
            st.error("Invalid Track Name. Check settings.")

# --- 7. ANALYTICS ---
if not history.empty:
    st.divider()
    st.header("📈 AI Evolution Metrics")
    
    valid = history.copy()
    if 'Predicted' in valid.columns and 'Actual' in valid.columns:
         predictions_exist = valid[valid['Predicted'].notna() & (valid['Predicted'] != "N/A")]
         if not predictions_exist.empty:
             predictions_exist['Is_Correct'] = (predictions_exist['Predicted'] == predictions_exist['Actual']).astype(int)
             
             c_metrics, c_stats = st.columns([1, 2])
             with c_metrics:
                 st.metric("Global Accuracy", f"{(predictions_exist['Is_Correct'].mean()*100):.1f}%")

    st.subheader("🧠 Learned Track Geometry")
    if 'Visible_Track' in valid.columns and 'Visible_Segment_%' in valid.columns:
        valid['Visible_Segment_%'] = pd.to_numeric(valid['Visible_Segment_%'], errors='coerce')
        stats = valid.groupby('Visible_Track')['Visible_Segment_%'].agg(['mean', 'std', 'count'])
        stats.columns = ['Avg Length %', 'Volatility', 'Races']
        st.dataframe(stats.style.format("{:.1f}"), use_container_width=True)

    with st.expander("🔍 Inspect Database (Full Context)"):
        st.dataframe(history.sort_index(ascending=False), use_container_width=True)
