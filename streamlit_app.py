import streamlit as st
import pandas as pd
import numpy as np
import os
from streamlit_extras.grid import grid

# --- 1. DATA & CONFIG ---
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
st.set_page_config(layout="wide", page_title="AI Race Predictor Pro", page_icon="🏎️")

# --- 1. DEFINE VALID TRACKS ONCE ---
VALID_TRACKS = list(SPEED_DATA["Car"].keys())

# --- 2. DATA LOADING & AUTOMATIC REPAIR ---
if os.path.exists(CSV_FILE):
    try:
        history = pd.read_csv(CSV_FILE)
        
        # Standardize naming immediately
        rename_map = {
            'Predicted_Winner': 'Predicted',
            'Actual_Winner': 'Actual',
            'Visible_%': 'Visible_Segment_%',
            'Visible_Lane_Length (%)': 'Visible_Segment_%'
        }
        history = history.rename(columns=rename_map)

        # CRITICAL FIX: If 'Visible_Track' is missing (KeyError), it means the CSV is corrupt.
        # We check if it exists; if not, we try to find it or reset.
        if 'Visible_Track' not in history.columns:
            st.warning("⚠️ Data structure corrupted. Attempting to repair...")
            # If the CSV has data but no headers, we'll skip it to prevent further errors
            history = pd.DataFrame(columns=["Visible_Track", "Visible_Segment_%", "Hidden_1_Track", 
                                            "Hidden_1_Len", "Hidden_2_Track", "Hidden_2_Len", 
                                            "Predicted", "Actual"])
        else:
            # 1. REMOVE POLLUTION: Delete rows where track is "20.0", "40.0", etc.
            history = history[history['Visible_Track'].isin(VALID_TRACKS)]
            
            # 2. FIX NUMERICS: Ensure lengths are actual numbers
            history['Visible_Segment_%'] = pd.to_numeric(history['Visible_Segment_%'], errors='coerce')
            history = history.dropna(subset=['Visible_Segment_%'])
            
    except Exception as e:
        history = pd.DataFrame()
else:
    history = pd.DataFrame()

# --- 3. UPDATED SAVE LOGIC (PREVENT POLLUTION) ---
# Use this block inside your "if submitted:" section
if submitted:
    last_probs = st.session_state.get('last_probs', {})
    predicted = max(last_probs, key=last_probs.get) if last_probs else "N/A"
    
    # Force data types to prevent "20.0" from slipping into the wrong column
    log_entry = {
        "Visible_Track": str(v_track), 
        "Visible_Segment_%": float(v_len),
        "Hidden_1_Track": str(h1_t),
        "Hidden_1_Len": float(h1_l),
        "Hidden_2_Track": str(h2_t),
        "Hidden_2_Len": float(h2_l),
        "Predicted": str(predicted),
        "Actual": str(winner)
    }
    
    # Save gatekeeper: Only save if the track is real
    if log_entry["Visible_Track"] in VALID_TRACKS:
        new_row = pd.DataFrame([log_entry])
        new_row.to_csv(CSV_FILE, mode='a', header=not os.path.exists(CSV_FILE), index=False)
        st.toast("Telemetry Synced!", icon="⚡")
        st.rerun()
# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("🚦 Race Setup")
    v_track = st.selectbox("Visible Track", list(SPEED_DATA["Car"].keys()))
    v_lane = st.radio("Active Lane", [1, 2, 3], horizontal=True)
    st.divider()
    c1 = st.selectbox("Vehicle 1 (Top)", list(SPEED_DATA.keys()), index=8)
    c2 = st.selectbox("Vehicle 2 (Mid)", list(SPEED_DATA.keys()), index=7)
    c3 = st.selectbox("Vehicle 3 (Bot)", list(SPEED_DATA.keys()), index=5)
    predict_btn = st.button("🚀 EXECUTE PREDICTION", type="primary", use_container_width=True)

# --- 5. MAIN INTERFACE ---
st.title("🏎️ AI RACE PREDICTOR PRO")

if predict_btn:
    probs = run_simulation_vectorized(c1, c2, c3, v_track, v_lane, history)
    st.session_state['last_probs'] = probs
    st.session_state['last_vehicles'] = [c1, c2, c3]
    
    m_grid = grid(3, vertical_align="center")
    for veh, val in probs.items():
        m_grid.metric(veh, f"{val:.1f}%")

    st.subheader("🚨 Strategic Risk")
    gap = max(probs.values()) - sorted(probs.values())[-2]
    if gap > 30: st.success("🏁 HIGH CONFIDENCE PREDICTION")
    else: st.warning("⚠️ VOLATILE RACE: Learning patterns...")

# --- 6. POST-RACE TELEMETRY ---
st.divider()
st.subheader("📝 POST-RACE TELEMETRY")
logger_vehicles = st.session_state.get('last_vehicles', [c1, c2, c3])

with st.form("logger_form", clear_on_submit=True):
    r1_c1, r1_c2 = st.columns(2)
    with r1_c1: winner = st.selectbox("Actual Winner", logger_vehicles)
    with r1_c2: v_len = st.number_input("Visible Segment Length %", 5, 95, 33)
    
    r2_c1, r2_c2 = st.columns(2)
    with r2_c1: h1_t = st.selectbox("Hidden Track 1 Type", list(SPEED_DATA["Car"].keys()))
    with r2_c2: h1_l = st.number_input("Hidden Segment 1 Length %", 5, 95, 33)
    
    r3_c1, r3_c2 = st.columns(2)
    with r3_c1: h2_t = st.selectbox("Hidden Track 2 Type", list(SPEED_DATA["Car"].keys()))
    with r3_c2: h2_l = st.number_input("Hidden Segment 2 Length %", 5, 95, 34)

    if st.form_submit_button("💾 SYNC TELEMETRY TO AI BRAIN", use_container_width=True):
        last_probs = st.session_state.get('last_probs', {})
        predicted = max(last_probs, key=last_probs.get) if last_probs else "N/A"
        
        log_entry = {
            "Visible_Track": v_track,
            "Visible_Segment_%": float(v_len),
            "Hidden_1_Track": h1_t, "Hidden_1_Len": float(h1_l),
            "Hidden_2_Track": h2_t, "Hidden_2_Len": float(h2_l),
            "Predicted": predicted, "Actual": winner
        }
        pd.DataFrame([log_entry]).to_csv(CSV_FILE, mode='a', header=not os.path.exists(CSV_FILE), index=False)
        st.rerun()
# --- UPDATE IN SECTION 6 (The Sync Button) ---
if submitted:
    last_probs = st.session_state.get('last_probs', {})
    predicted = max(last_probs, key=last_probs.get) if last_probs else "N/A"
    
    # DOUBLE CHECK: Ensure we are saving the 'v_track' string, not a number
    log_entry = {
        "Visible_Track": str(v_track),  # Explicitly save the name (e.g., "Desert")
        "Visible_Segment_%": float(v_len),
        "Hidden_1_Track": str(h1_t),
        "Hidden_1_Len": float(h1_l),
        "Hidden_2_Track": str(h2_t),
        "Hidden_2_Len": float(h2_l),
        "Predicted": str(predicted),
        "Actual": str(winner)
    }
    
    # Only save if the track is valid
    if log_entry["Visible_Track"] in VALID_TRACKS:
        pd.DataFrame([log_entry]).to_csv(CSV_FILE, mode='a', header=not os.path.exists(CSV_FILE), index=False)
        st.toast("Telemetry Synced! AI Brain Cleaned.", icon="⚡")
        st.rerun()
    else:
        st.error(f"Invalid track detected: {v_track}. Data not saved.")
# --- 7. ANALYTICS ---
if not history.empty:
    st.divider()
    st.header("📈 AI Learning Analytics")
    
    # Filter valid rows for accuracy
    if 'Predicted' in history.columns and 'Actual' in history.columns:
        valid_history = history[(history['Predicted'] != "N/A") & (history['Actual'].notna())].copy()
        
        if not valid_history.empty:
            valid_history['Is_Correct'] = (valid_history['Predicted'] == valid_history['Actual']).astype(int)
            
            c_acc, c_learned = st.columns([1, 2])
            with c_acc:
                st.metric("Global Accuracy", f"{(valid_history['Is_Correct'].mean()*100):.1f}%")
                st.write("**Accuracy Heatmap**")
                heatmap = valid_history.groupby('Visible_Track')['Is_Correct'].mean() * 100
                st.dataframe(heatmap.to_frame('Acc %').style.background_gradient(cmap='RdYlGn'), use_container_width=True)

            with c_learned:
                st.write("**Learned Track Geometry (Last 20 Races)**")
                # Ensure Visible_Segment_% is treated as numeric here too
                valid_history['Visible_Segment_%'] = pd.to_numeric(valid_history['Visible_Segment_%'], errors='coerce')
                learned_stats = valid_history.groupby('Visible_Track')['Visible_Segment_%'].agg(['mean', 'std', 'count'])
                learned_stats.columns = ['Avg Length %', 'Volatility', 'Races']
                st.dataframe(learned_stats.style.format("{:.1f}").background_gradient(cmap='Blues', subset=['Avg Length %']), use_container_width=True)

    with st.expander("🔍 View Full Telemetry Log"):
        st.dataframe(history.sort_index(ascending=False), use_container_width=True)
        # Add this inside your 'with st.expander("🔍 View Full Telemetry Log"):' block
if st.button("🧹 PURGE INVALID TRACK DATA"):
    # Keep only rows where the track name is a real track type
    valid_tracks = list(SPEED_DATA["Car"].keys())
    cleaned_history = history[history['Visible_Track'].isin(valid_tracks)]
    cleaned_history.to_csv(CSV_FILE, index=False)
    st.success("Invalid tracks (like '20.0') removed! Reloading...")
    st.rerun()

if st.button("🗑️ RESET ALL LEARNING (DELETE CSV)"):
    if os.path.exists(CSV_FILE):
        os.remove(CSV_FILE)
        st.warning("AI Brain wiped clean. Starting fresh!")
        st.rerun()
