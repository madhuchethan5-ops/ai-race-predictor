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

CSV_FILE = 'race_history.csv'
VALID_TRACKS = list(SPEED_DATA["Car"].keys())
st.set_page_config(layout="wide", page_title="AI Race Predictor Pro", page_icon="🏎️")

# --- 2. SELF-HEALING DATA LOADER ---
# Detects corruption, standardizes columns, and filters bad data on startup.
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
        
        # 2. Gatekeeper: Remove rows where Track Name is corrupted (e.g., "20.0")
        if 'Visible_Track' in df.columns:
            df = df[df['Visible_Track'].isin(VALID_TRACKS)]
            
        # 3. Type Enforcement: Ensure percentages are strictly numeric
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
    all_terrains = list(SPEED_DATA["Car"].keys())
    
    # --- A. LEARNING PHASE ---
    # Default Uncertainty (Bayesian Prior): We assume 33% length with high variance (unknown)
    avg_vis = 0.33
    vis_std = 0.12 # High standard deviation = "AI is unsure"
    
    # If we have valid history for this specific track, update the Prior
    if not history_df.empty and 'Visible_Segment_%' in history_df.columns:
        # Filter for THIS track type only (e.g., only look at previous "Desert" races)
        track_data = history_df[history_df['Visible_Track'] == visible_t].tail(20)
        
        if not track_data.empty:
            # LEARN: Update the average length based on history
            avg_vis = track_data['Visible_Segment_%'].mean() / 100
            
            # ADAPT: If data is consistent, reduce variance. If volatile, keep variance high.
            if len(track_data) > 1:
                vis_std = max(0.04, track_data['Visible_Segment_%'].std() / 100)

    # --- B. MONTE CARLO EXECUTION ---
    # 1. Generate 5,000 track scenarios using the Learned Distribution
    vis_lens = np.clip(np.random.normal(avg_vis, vis_std, iterations), 0.05, 0.95)
    
    # 2. Randomize Hidden Segments (ensuring Total = 100%)
    remaining = 1.0 - vis_lens
    h1_ratios = np.random.uniform(0.1, 0.9, iterations)
    h1_lens = remaining * h1_ratios
    h2_lens = remaining - h1_lens

    # 3. Randomize Hidden Terrain Types
    seg_terrains = np.random.choice(all_terrains, size=(iterations, 3))
    seg_terrains[:, visible_l-1] = visible_t # The visible lane is fixed

    # 4. Run Physics Simulation
    results = {}
    for v in vehicles:
        # Vectorized Speed Lookup
        speed_lookup = np.vectorize(SPEED_DATA[v].get)(seg_terrains)
        
        # Apply 2% Driver Skill Variance (Luck Factor)
        noise = np.random.normal(1.0, 0.02, (iterations, 3))
        noisy_speeds = speed_lookup * noise
        
        # Time = Distance / Speed
        times = (vis_lens/noisy_speeds[:, 0]) + (h1_lens/noisy_speeds[:, 1]) + (h2_lens/noisy_speeds[:, 2])
        results[v] = times

    # 5. Determine Winners
    winners = np.argmin(np.array([results[v] for v in vehicles]), axis=0)
    counts = pd.Series(winners).value_counts(normalize=True).sort_index() * 100
    
    return {vehicles[i]: counts.get(i, 0) for i in range(3)}

# --- 4. CONTROL PANEL (SIDEBAR) ---
with st.sidebar:
    st.header("🚦 Race Setup")
    v_track = st.selectbox("Visible Track", list(SPEED_DATA["Car"].keys()))
    v_lane = st.radio("Active Lane", [1, 2, 3], horizontal=True)
    st.divider()
    c1 = st.selectbox("Vehicle 1 (Top)", list(SPEED_DATA.keys()), index=8)
    c2 = st.selectbox("Vehicle 2 (Mid)", list(SPEED_DATA.keys()), index=7)
    c3 = st.selectbox("Vehicle 3 (Bot)", list(SPEED_DATA.keys()), index=5)
    
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
    # Run the Adaptive Simulation
    probs = run_simulation(c1, c2, c3, v_track, v_lane, history)
    
    # Store results for logging
    st.session_state['last_probs'] = probs
    st.session_state['last_vehicles'] = [c1, c2, c3]
    
    # Display Results
    m_grid = grid(3, vertical_align="center")
    for veh, val in probs.items():
        m_grid.metric(veh, f"{val:.1f}%")

    # Strategic Risk Assessment
    st.subheader("🚨 Strategic Confidence")
    gap = max(probs.values()) - sorted(probs.values())[-2]
    
    if gap > 35:
        st.success(f"🏁 HIGH CONFIDENCE: {max(probs, key=probs.get)} has a strong statistical advantage.")
    elif gap > 15:
        st.warning("⚠️ MODERATE RISK: Hidden segments will play a major role.")
    else:
        st.error("⚡ EXTREME VOLATILITY: Too close to call. Outcome is random.")

# --- 6. TELEMETRY LOGGING (TRAINING INPUT) ---
st.divider()
st.subheader("📝 POST-RACE TELEMETRY (TRAINING DATA)")
logger_vehicles = st.session_state.get('last_vehicles', [c1, c2, c3])

with st.form("logger_form", clear_on_submit=True):
    c_a, c_b = st.columns(2)
    with c_a: winner = st.selectbox("Actual Winner", logger_vehicles)
    with c_b: v_len = st.number_input("Visible Segment Length %", 0.0, 100.0, 33.0, step=1.0)
    
    c_c, c_d = st.columns(2)
    with c_c: h1_t = st.selectbox("Hidden 1 Type", list(SPEED_DATA["Car"].keys()))
    with c_d: h1_l = st.number_input("Hidden 1 Length %", 0.0, 100.0, 33.0, step=1.0)
    
    c_e, c_f = st.columns(2)
    with c_e: h2_t = st.selectbox("Hidden 2 Type", list(SPEED_DATA["Car"].keys()))
    with c_f: h2_l = st.number_input("Hidden 2 Length %", 0.0, 100.0, 34.0, step=1.0)

    if st.form_submit_button("💾 FEED DATA TO AI", use_container_width=True):
        last_probs = st.session_state.get('last_probs', {})
        predicted = max(last_probs, key=last_probs.get) if last_probs else "N/A"
        
        # TYPE-SAFE LOGGING (Prevents database corruption)
        log_entry = {
            "Visible_Track": str(v_track),
            "Visible_Segment_%": float(v_len),
            "Hidden_1_Track": str(h1_t), "Hidden_1_Len": float(h1_l),
            "Hidden_2_Track": str(h2_t), "Hidden_2_Len": float(h2_l),
            "Predicted": str(predicted), "Actual": str(winner)
        }
        
        # Gatekeeper: Only save valid track data
        if log_entry["Visible_Track"] in VALID_TRACKS:
            pd.DataFrame([log_entry]).to_csv(CSV_FILE, mode='a', header=not os.path.exists(CSV_FILE), index=False)
            st.toast("AI Calibration Updated!", icon="🧠")
            st.rerun()
        else:
            st.error("Invalid Track detected. Data discarded to protect AI integrity.")

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
                st.caption("Accuracy improves as you log more races.")
                
                # Accuracy Heatmap
                st.write("**Performance by Track**")
                heatmap = valid.groupby('Visible_Track')['Is_Correct'].mean() * 100
                st.dataframe(heatmap.to_frame("Acc %").style.background_gradient(cmap="RdYlGn", vmin=0, vmax=100), use_container_width=True)

            with c_stats:
                st.write("**🧠 Learned Track Geometry (The AI Brain)**")
                st.info("The AI uses these 'Avg Lengths' to simulate the hidden segments.")
                
                # Show exactly what numbers the AI is using for simulation
                stats = valid.groupby('Visible_Track')['Visible_Segment_%'].agg(['mean', 'std', 'count'])
                stats.columns = ['Avg Length %', 'Volatility', 'Races Logged']
                st.dataframe(stats.style.format("{:.1f}"), use_container_width=True)

    with st.expander("🔍 Inspect Raw Database"):
        st.dataframe(history.sort_index(ascending=False), use_container_width=True)
# --- NEW: VISUAL PERFORMANCE GRAPH ---
            st.subheader("📉 AI Learning Curve (Rolling 10-Race Average)")
            
            # 1. Create a "1" for Correct, "0" for Wrong
            valid['Is_Correct'] = (valid['Predicted'] == valid['Actual']).astype(int)
            
            # 2. Calculate Rolling Average (The "Trend")
            # This smooths out the line so you see the general direction of improvement
            valid['Accuracy_Trend'] = valid['Is_Correct'].rolling(window=10, min_periods=1).mean() * 100
            
            # 3. Plot the Line Chart
            st.line_chart(valid['Accuracy_Trend'], color="#00FF00", height=250)
            
            st.caption("✅ **Green Line Rising:** The AI is successfully learning your track patterns.")
            st.caption("🔻 **Line Flat/Falling:** The track is unpredictable or data is inconsistent.")
