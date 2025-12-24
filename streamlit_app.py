import streamlit as st
import pandas as pd
import numpy as np
import os
from streamlit_extras.grid import grid

# --- 1. CONFIGURATION & PHYSICS ---
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

# --- 2. ROBUST DATA LOADER (SELF-HEALING) ---
def load_data():
    if not os.path.exists(CSV_FILE):
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(CSV_FILE)
        
        # 1. Standardize Legacy Column Names
        rename_map = {
            'Predicted_Winner': 'Predicted', 'Actual_Winner': 'Actual',
            'Visible_%': 'Visible_Segment_%', 'Visible_Lane_Length (%)': 'Visible_Segment_%'
        }
        df = df.rename(columns=rename_map)
        
        # 2. Gatekeeper: Remove rows where Track Name is not valid (fixes "20.0" bug)
        if 'Visible_Track' in df.columns:
            df = df[df['Visible_Track'].isin(VALID_TRACKS)]
        
        # 3. Type Safety: Ensure percentages are numbers
        if 'Visible_Segment_%' in df.columns:
            df['Visible_Segment_%'] = pd.to_numeric(df['Visible_Segment_%'], errors='coerce')
            df = df.dropna(subset=['Visible_Segment_%'])
            
        return df
    except Exception as e:
        st.error(f"⚠️ History file corrupted. Starting fresh session. Error: {e}")
        return pd.DataFrame()

history = load_data()

# --- 3. VECTORIZED SIMULATION ENGINE (THE BRAIN) ---
def run_simulation(v1, v2, v3, visible_t, visible_l, history_df, iterations=5000):
    vehicles = [v1, v2, v3]
    all_terrains = list(SPEED_DATA["Car"].keys())
    
    # --- LEARNING LOGIC ---
    # Default "Initial Knowledge" (Fallback)
    avg_vis = 0.33
    vis_std = 0.08 # High variance = "I'm not sure"
    
    # If we have history, use it to sharpen the prediction
    if not history_df.empty and 'Visible_Segment_%' in history_df.columns:
        match = history_df[history_df['Visible_Track'] == visible_t].tail(20) # Look at last 20 races
        if not match.empty:
            avg_vis = match['Visible_Segment_%'].mean() / 100
            # If we have >1 race, calculate real volatility. Else keep default uncertainty.
            if len(match) > 1:
                vis_std = max(0.04, match['Visible_Segment_%'].std() / 100)

    # --- MONTE CARLO EXECUTION ---
    # Generate 5000 scenarios based on the learned distribution
    vis_lens = np.clip(np.random.normal(avg_vis, vis_std, iterations), 0.05, 0.95)
    
    # Randomize the hidden segments (Dirichlet distribution ensures they sum to 1.0 - vis)
    remaining = 1.0 - vis_lens
    h1_ratios = np.random.uniform(0.1, 0.9, iterations)
    h1_lens = remaining * h1_ratios
    h2_lens = remaining - h1_lens

    # Randomize hidden terrain types
    seg_terrains = np.random.choice(all_terrains, size=(iterations, 3))
    seg_terrains[:, visible_l-1] = visible_t # Hardcode the known visible track

    results = {}
    for v in vehicles:
        # Vectorized Speed Lookup
        speed_lookup = np.vectorize(SPEED_DATA[v].get)(seg_terrains)
        
        # Apply 2% Driver Variance (Luck Factor)
        noise = np.random.normal(1.0, 0.02, (iterations, 3))
        noisy_speeds = speed_lookup * noise
        
        # Calculate Time: T = Distance / Speed
        times = (vis_lens/noisy_speeds[:, 0]) + (h1_lens/noisy_speeds[:, 1]) + (h2_lens/noisy_speeds[:, 2])
        results[v] = times

    # Determine Winners
    winners = np.argmin(np.array([results[v] for v in vehicles]), axis=0)
    counts = pd.Series(winners).value_counts(normalize=True).sort_index() * 100
    
    # Return formatted dict
    return {vehicles[i]: counts.get(i, 0) for i in range(3)}

# --- 4. UI: SIDEBAR ---
with st.sidebar:
    st.header("🚦 Race Setup")
    v_track = st.selectbox("Visible Track", list(SPEED_DATA["Car"].keys()))
    v_lane = st.radio("Active Lane", [1, 2, 3], horizontal=True)
    st.divider()
    c1 = st.selectbox("Vehicle 1 (Top)", list(SPEED_DATA.keys()), index=8)
    c2 = st.selectbox("Vehicle 2 (Mid)", list(SPEED_DATA.keys()), index=7)
    c3 = st.selectbox("Vehicle 3 (Bot)", list(SPEED_DATA.keys()), index=5)
    
    # The Action Button
    predict_btn = st.button("🚀 EXECUTE PREDICTION", type="primary", use_container_width=True)
    
    st.divider()
    with st.expander("🗑️ Data Reset"):
        if st.button("Wipe All History"):
            if os.path.exists(CSV_FILE):
                os.remove(CSV_FILE)
                st.rerun()

# --- 5. UI: MAIN DASHBOARD ---
st.title("🏎️ AI RACE PREDICTOR PRO")

if predict_btn:
    probs = run_simulation(c1, c2, c3, v_track, v_lane, history)
    st.session_state['last_probs'] = probs
    st.session_state['last_vehicles'] = [c1, c2, c3]
    
    # 1. High-Level Metrics
    m_grid = grid(3, vertical_align="center")
    for veh, val in probs.items():
        m_grid.metric(veh, f"{val:.1f}%")

    # 2. Risk Analysis
    st.subheader("🚨 Strategic Risk")
    gap = max(probs.values()) - sorted(probs.values())[-2]
    
    if gap > 35:
        st.success(f"🏁 HIGH CONFIDENCE: {max(probs, key=probs.get)} is the statistical favorite.")
    elif gap > 15:
        st.warning("⚠️ MODERATE RISK: Hidden segments will decide the outcome.")
    else:
        st.error("⚡ EXTREME VOLATILITY: Too close to call. Avoid high stakes.")

# --- 6. TELEMETRY LOGGING (TYPE-SAFE) ---
st.divider()
st.subheader("📝 POST-RACE TELEMETRY")
logger_vehicles = st.session_state.get('last_vehicles', [c1, c2, c3])

with st.form("logger_form", clear_on_submit=True):
    c_a, c_b = st.columns(2)
    with c_a: winner = st.selectbox("Actual Winner", logger_vehicles)
    with c_b: v_len = st.number_input("Visible Segment Length %", 5.0, 95.0, 33.0, step=1.0)
    
    c_c, c_d = st.columns(2)
    with c_c: h1_t = st.selectbox("Hidden 1 Type", list(SPEED_DATA["Car"].keys()))
    with c_d: h1_l = st.number_input("Hidden 1 Length %", 5.0, 95.0, 33.0, step=1.0)
    
    c_e, c_f = st.columns(2)
    with c_e: h2_t = st.selectbox("Hidden 2 Type", list(SPEED_DATA["Car"].keys()))
    with c_f: h2_l = st.number_input("Hidden 2 Length %", 5.0, 95.0, 34.0, step=1.0)

    if st.form_submit_button("💾 SYNC TO AI BRAIN", use_container_width=True):
        last_probs = st.session_state.get('last_probs', {})
        predicted = max(last_probs, key=last_probs.get) if last_probs else "N/A"
        
        # STRICT DATA SAVING
        log_entry = {
            "Visible_Track": str(v_track),
            "Visible_Segment_%": float(v_len),
            "Hidden_1_Track": str(h1_t), "Hidden_1_Len": float(h1_l),
            "Hidden_2_Track": str(h2_t), "Hidden_2_Len": float(h2_l),
            "Predicted": str(predicted), "Actual": str(winner)
        }
        
        # Valid Track Gatekeeper
        if log_entry["Visible_Track"] in VALID_TRACKS:
            pd.DataFrame([log_entry]).to_csv(CSV_FILE, mode='a', header=not os.path.exists(CSV_FILE), index=False)
            st.toast("Telemetry Saved & AI Calibrated!", icon="✅")
            st.rerun()
        else:
            st.error("Invalid Track Name detected. Data discarded.")

# --- 7. ANALYTICS & LEARNING CURVE ---
if not history.empty:
    st.divider()
    st.header("📈 AI Learning Analytics")
    
    # Only calculate stats on valid rows
    if 'Predicted' in history.columns and 'Actual' in history.columns:
        valid = history[history['Predicted'] != "N/A"].copy()
        
        if not valid.empty:
            valid['Is_Correct'] = (valid['Predicted'] == valid['Actual']).astype(int)
            
            c_metrics, c_stats = st.columns([1, 2])
            with c_metrics:
                acc = valid['Is_Correct'].mean() * 100
                st.metric("Global Accuracy", f"{acc:.1f}%")
                
                # Heatmap
                st.write("**Accuracy Heatmap**")
                heatmap = valid.groupby('Visible_Track')['Is_Correct'].mean() * 100
                st.dataframe(heatmap.to_frame("Acc %").style.background_gradient(cmap="RdYlGn", vmin=0, vmax=100), use_container_width=True)

            with c_stats:
                st.write(f"**Learned Geometry (Last 20 Races)**")
                # Show what the AI thinks the average length is
                stats = valid.groupby('Visible_Track')['Visible_Segment_%'].agg(['mean', 'std', 'count'])
                stats.columns = ['Avg Length %', 'Volatility', 'Races']
                st.dataframe(stats.style.format("{:.1f}"), use_container_width=True)

    with st.expander("🔍 View Raw Data"):
        st.dataframe(history.sort_index(ascending=False), use_container_width=True)
