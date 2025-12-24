import streamlit as st
import pandas as pd
import numpy as np
import os

# --- STEP 1: YOUR EXACT SPEED DATA ---
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

# --- FILE MANAGEMENT: Create/Load History ---
CSV_FILE = 'race_history.csv'
if not os.path.exists(CSV_FILE):
    # Initialize file if it doesn't exist
    df_init = pd.DataFrame(columns=["V1", "V2", "V3", "Visible_Track", "Lane", "Predicted_Winner", "Actual_Winner"])
    df_init.to_csv(CSV_FILE, index=False)

# --- STEP 2: APP LAYOUT ---
st.set_page_config(page_title="Race Predictor & Logger", page_icon="🏎️")
st.title("🏎️ AI Race Predictor & Logger")

# Sidebar for Inputs
with st.sidebar:
    st.header("1. Race Setup")
    v1 = st.selectbox("Vehicle 1", list(SPEED_DATA.keys()), index=8)
    v2 = st.selectbox("Vehicle 2", list(SPEED_DATA.keys()), index=7)
    v3 = st.selectbox("Vehicle 3", list(SPEED_DATA.keys()), index=5)
    
    st.divider()
    visible_track = st.selectbox("Visible Track", ["Expressway", "Desert", "Dirt", "Potholes", "Bumpy", "Highway"])
    visible_lane = st.radio("Visible Lane Position", [1, 2, 3])

# --- STEP 3: PREDICTION ENGINE ---
def run_simulation(v1, v2, v3, visible_t, visible_l):
    wins = {v1: 0, v2: 0, v3: 0}
    iterations = 2000
    all_terrains = list(SPEED_DATA["Car"].keys())

    for _ in range(iterations):
        lengths = np.random.dirichlet(np.ones(3), size=1)[0]
        t1 = visible_t if visible_l == 1 else np.random.choice(all_terrains)
        t2 = visible_t if visible_l == 2 else np.random.choice(all_terrains)
        t3 = visible_t if visible_l == 3 else np.random.choice(all_terrains)
        tracks = [t1, t2, t3]
        
        times = {}
        for v in [v1, v2, v3]:
            time = sum([(lengths[i] * 1000) / SPEED_DATA[v][tracks[i]] for i in range(3)])
            times[v] = time
        
        winner = min(times, key=times.get)
        wins[winner] += 1
    
    return {k: (v / iterations) * 100 for k, v in wins.items()}

# --- MAIN PAGE: PREDICTION ---
st.markdown("### 📊 Live Prediction")
if st.button("🚀 Predict Winner", type="primary"):
    with st.spinner('Running Monte Carlo Simulation...'):
        probs = run_simulation(v1, v2, v3, visible_track, visible_lane)
    
    likely_winner = max(probs, key=probs.get)
    st.session_state['last_prediction'] = likely_winner  # Save for logging
    
    st.success(f"🏆 Predicted Winner: **{likely_winner}**")
    
    for vehicle, percent in probs.items():
        st.write(f"**{vehicle}**: {percent:.1f}%")
        st.progress(int(percent))

st.divider()

# --- STEP 4: DATA LOGGING (THE NEW PART) ---
st.markdown("### 📝 Log Results (Build Your Database)")
st.info("After the race is over, select the actual winner below and click 'Save'.")

# Dropdown to pick who actually won
actual_winner = st.selectbox("Who actually won?", [v1, v2, v3], key="winner_select")

if st.button("💾 Save Result to Database"):
    # Load current history
    df_history = pd.read_csv(CSV_FILE)
    
    # Create new row
    new_data = {
        "V1": v1, "V2": v2, "V3": v3,
        "Visible_Track": visible_track,
        "Lane": visible_lane,
        "Predicted_Winner": st.session_state.get('last_prediction', "N/A"),
        "Actual_Winner": actual_winner
    }
    
    # Save to CSV
    df_new = pd.DataFrame([new_data])
    df_history = pd.concat([df_history, df_new], ignore_index=True)
    df_history.to_csv(CSV_FILE, index=False)
    
    st.success(f"✅ Saved! Database now has {len(df_history)} races.")

# Show History
with st.expander("📂 View Race History"):
    if os.path.exists(CSV_FILE):
        st.dataframe(pd.read_csv(CSV_FILE).tail(10)) # Show last 10 races
