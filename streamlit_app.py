import streamlit as st
import pandas as pd
import numpy as np
import os
from streamlit_extras.grid import grid

# --- 1. CONFIGURATION & PHYSICS ENGINE ---
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

# --- MASTER SORTED LISTS (DEFINED GLOBALLY) ---
# This forces the order to be A-Z for the entire app.
VEHICLE_OPTIONS = sorted(list(SPEED_DATA.keys()))
TRACK_OPTIONS = sorted(list(SPEED_DATA["Car"].keys()))

CSV_FILE = 'race_history.csv'
VALID_TRACKS = TRACK_OPTIONS
st.set_page_config(layout="wide", page_title="AI Race Predictor Pro", page_icon="🏎️")

# --- 2. SELF-HEALING DATA LOADER ---
def load_clean_history():
    if not os.path.exists(CSV_FILE):
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(CSV_FILE)
        
        # 1. Standardize Names
        rename_map = {
            'Predicted_Winner': 'Predicted', 'Actual_Winner': 'Actual',
            'Visible_%': 'Visible_Segment_%', 'Visible_Lane_Length (%)': 'Visible_Segment_%'
        }
        df = df.rename(columns=rename_map)
        
        # 2. Gatekeeper: Remove rows where Track Name is corrupted
        if 'Visible_Track' in df.columns:
            df = df[df['Visible_Track'].isin(VALID_TRACKS)]
            
        # 3. Type Enforcement: Ensure percentages are numeric
        if 'Visible_Segment_%' in df.columns:
            df['Visible_Segment_%'] = pd.to_numeric(df['Visible_Segment_%'], errors='coerce')
            df = df.dropna(subset=['Visible_Segment_%'])
            
        return df
    except Exception:
        return pd.DataFrame()

history = load_clean_history()

# --- 3. ADAPTIVE BAYESIAN SIMULATION ENGINE ---
def run_simulation(v1, v2, v3, visible_t, visible_l, history_df, iterations=5000):
    vehicles = [v1, v2, v3]
    all_terrains = TRACK_OPTIONS # Use the sorted list
    
    # --- LEARNING PHASE ---
    avg_vis = 0.33
    vis_std = 0.12 
    
    if not history_df.empty and 'Visible_Segment_%' in history_df.columns:
        track_data = history_df[history_df['Visible_Track'] == visible_t].tail(20)
        
        if not track_data.empty:
            avg_vis = track_data['Visible_Segment_%'].mean() / 100
            if len(track_data) > 1:
                vis_std = max(0.04, track_data['Visible_Segment_%'].std() / 100)

    # --- MONTE CARLO EXECUTION ---
    vis_lens = np.clip(np.random.normal(avg_vis, vis_std, iterations), 0.05, 0.95)
    remaining = 1.0 - vis_lens
    h1_ratios = np.random.uniform(0.1, 0.9, iterations)
    h1_lens = remaining * h1_ratios
    h2_lens = remaining - h1_lens

    seg_terrains = np.random.choice(all_terrains, size=(iterations, 3))
    seg_terrains[:, visible_l-1] = visible_t 

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

# --- 4. CONTROL PANEL (USING MASTER SORTED LIST) ---
with st.sidebar:
    st.header("🚦 Race Setup")
    
    # Track Selection (Uses Sorted List)
    v_track = st.selectbox("Visible Track", TRACK_OPTIONS)
    v_lane = st.radio("Active Lane", [1, 2, 3], horizontal=True)
    st.divider()
    
    # Vehicle Selection (Uses Sorted List)
    # We use .index() to find the correct default regardless of sort order
    c1 = st.selectbox("Vehicle 1 (Top)", VEHICLE_OPTIONS, index=VEHICLE_OPTIONS.index("Supercar"))
    c2 = st.selectbox("Vehicle 2 (Mid)", VEHICLE_OPTIONS, index=VEHICLE_OPTIONS.index("Sports Car"))
    c3 = st.selectbox("Vehicle 3 (Bot)", VEHICLE_OPTIONS, index=VEHICLE_OPTIONS.index("Car"))
    
    predict_btn = st.button("🚀 PREDICT OUTCOME", type="primary", use_container_width=True)
    
    st.divider()
    with st.expander("🛠️ Data Tools"):
        if st.button("🗑️ Wipe Memory"):
            if os.path.exists(CSV_FILE):
                os.remove(CSV_FILE)
                st.rerun()

# --- 5. MAIN DASHBOARD ---
st.title("🏎️ AI RACE PREDICTOR PRO")

if predict_btn:
    probs = run_simulation(c1, c2, c3, v_track, v_lane, history)
    
    st.session_state['last_probs'] = probs
    st.session_state['last_vehicles'] = [c1, c2, c3]
    
    m_grid = grid(3, vertical_align="center")
    for veh, val in probs.items():
        m_grid.metric(veh, f"{val:.1f}%")

    st.subheader("🚨 Strategic Confidence")
    gap = max(probs.values()) - sorted(probs.values())[-2]
    
    if gap > 35:
        st.success(f"🏁 HIGH CONFIDENCE: {max(probs, key=probs.get)} is the statistical favorite.")
    elif gap > 15:
        st.warning("⚠️ MODERATE RISK: Hidden segments will play a major role.")
    else:
        st.error("⚡ EXTREME VOLATILITY: Too close to call.")

# --- 6. TELEMETRY LOGGING (USING MASTER SORTED LIST) ---
st.divider()
st.subheader("📝 POST-RACE TELEMETRY")
logger_vehicles = st.session_state.get('last_vehicles', [c1, c2, c3])

with st.form("logger_form", clear_on_submit=True):
    c_a, c_b = st.columns(2)
    with c_a: winner = st.selectbox("Actual Winner", logger_vehicles)
    with c_b: v_len = st.number_input("Visible Segment Length %", 0.0, 100.0, 33.0, step=1.0)
    
    c_c, c_d = st.columns(2)
    # Using TRACK_OPTIONS ensures these are A-Z sorted
    with c_c: h1_t = st.selectbox("Hidden 1 Type", TRACK_OPTIONS)
    with c_d: h1_l = st.number_input("Hidden 1 Length %", 0.0, 100.0, 33.0, step=1.0)
    
    c_e, c_f = st.columns(2)
    # Using TRACK_OPTIONS ensures these are A-Z sorted
    with c_e: h2_t = st.selectbox("Hidden 2 Type", TRACK_OPTIONS)
    with c_f: h2_l = st.number_input("Hidden 2 Length %", 0.0, 100.0, 34.0, step=1.0)

    if st.form_submit_button("💾 FEED DATA TO AI", use_container_width=True):
        last_probs = st.session_state.get('last_probs', {})
        predicted = max(last_probs, key=last_probs.get) if last_probs else "N/A"
        
        log_entry = {
            "Visible_Track": str(v_track),
            "Visible_Segment_%": float(v_len),
            "Hidden_1_Track": str(h1_t), "Hidden_1_Len": float(h1_l),
            "Hidden_2_Track": str(h2_t), "Hidden_2_Len": float(h2_l),
            "Predicted": str(predicted), "Actual": str(winner)
        }
        
        if log_entry["Visible_Track"] in VALID_TRACKS:
            pd.DataFrame([log_entry]).to_csv(CSV_FILE, mode='a', header=not os.path.exists(CSV_FILE), index=False)
            st.toast("AI Calibration Updated!", icon="🧠")
            st.rerun()
        else:
            st.error("Invalid Track detected. Data discarded.")

# --- 7. REAL-TIME LEARNING ANALYTICS ---
if not history.empty:
    st.divider()
    st.header("📈 AI Evolution Metrics")
    
    if 'Predicted' in history.columns and 'Actual' in history.columns:
        valid = history[history['Predicted'] != "N/A"].copy()
        
        if not valid.empty:
            valid['Is_Correct'] = (valid['Predicted'] == valid['Actual']).astype(int)
            
            c_metrics, c_stats = st.columns([1, 2])
            with c_metrics:
                st.metric("Global Prediction Accuracy", f"{(valid['Is_Correct'].mean()*100):.1f}%")
                
                st.write("**Performance by Track**")
                heatmap = valid.groupby('Visible_Track')['Is_Correct'].mean() * 100
                st.dataframe(heatmap.to_frame("Acc %").style.background_gradient(cmap="RdYlGn", vmin=0, vmax=100), use_container_width=True)

            with c_stats:
                st.write("**📉 Learning Curve (Trend over last 10 races)**")
                valid['Accuracy_Trend'] = valid['Is_Correct'].rolling(window=10, min_periods=1).mean() * 100
                st.line_chart(valid['Accuracy_Trend'], color="#00FF00", height=200)

                st.write("**🧠 Learned Track Geometry**")
                stats = valid.groupby('Visible_Track')['Visible_Segment_%'].agg(['mean', 'std', 'count'])
                stats.columns = ['Avg Length %', 'Volatility', 'Races']
                st.dataframe(stats.style.format("{:.1f}"), use_container_width=True)

    with st.expander("🔍 Inspect Raw Database"):
        st.dataframe(history.sort_index(ascending=False), use_container_width=True)
