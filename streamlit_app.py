import streamlit as st
import pandas as pd
import numpy as np

# --- STEP 1: YOUR EXACT SPEED DATA ---
# These are the raw speed values (Higher = Faster)
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

# --- STEP 2: APP LAYOUT ---
st.set_page_config(page_title="Race Predictor", page_icon="🏎️")
st.title("🏎️ AI Race Predictor Pro")
st.markdown("### Enter Live Race Conditions")

# Sidebar for Inputs
with st.sidebar:
    st.header("Race Setup")
    
    # Dropdowns for Vehicles
    v1 = st.selectbox("Vehicle 1", list(SPEED_DATA.keys()), index=8) # Default: Supercar
    v2 = st.selectbox("Vehicle 2", list(SPEED_DATA.keys()), index=7) # Default: Sports Car
    v3 = st.selectbox("Vehicle 3", list(SPEED_DATA.keys()), index=5) # Default: Car
    
    st.divider()
    
    # Track Conditions
    visible_track = st.selectbox("Visible Track (Shown)", ["Expressway", "Desert", "Dirt", "Potholes", "Bumpy", "Highway"])
    visible_lane = st.radio("Which Lane is Visible?", [1, 2, 3])

# --- STEP 3: THE AI SIMULATION (Monte Carlo) ---
def run_simulation(v1, v2, v3, visible_t, visible_l):
    wins = {v1: 0, v2: 0, v3: 0}
    iterations = 2000  # Run 2,000 virtual races for accuracy
    
    # Get all possible tracks to choose from for hidden sections
    all_terrains = list(SPEED_DATA["Car"].keys())

    for _ in range(iterations):
        # 1. Randomize Lane Lengths (e.g., Lane 1=50%, Lane 2=30%, Lane 3=20%)
        # Dirichlet distribution ensures they sum to 1.0 (100%)
        lengths = np.random.dirichlet(np.ones(3), size=1)[0]
        
        # 2. Assign Tracks to Lanes
        # The visible lane is fixed. The other two are random guesses.
        t1 = visible_t if visible_l == 1 else np.random.choice(all_terrains)
        t2 = visible_t if visible_l == 2 else np.random.choice(all_terrains)
        t3 = visible_t if visible_l == 3 else np.random.choice(all_terrains)
        
        tracks = [t1, t2, t3]

        # 3. Calculate Race Time for each vehicle
        # Logic: Time = Distance / Speed
        # We assume Total Distance = 1000 units. 
        # So Lane Distance = Length * 1000.
        times = {}
        for v in [v1, v2, v3]:
            # Calculate time taken for each lane
            time_lane_1 = (lengths[0] * 1000) / SPEED_DATA[v][tracks[0]]
            time_lane_2 = (lengths[1] * 1000) / SPEED_DATA[v][tracks[1]]
            time_lane_3 = (lengths[2] * 1000) / SPEED_DATA[v][tracks[2]]
            
            # Total race time
            times[v] = time_lane_1 + time_lane_2 + time_lane_3
        
        # 4. Determine Winner (Lowest Time Wins)
        winner = min(times, key=times.get)
        wins[winner] += 1
    
    # Convert wins to percentages
    return {k: (v / iterations) * 100 for k, v in wins.items()}

# --- STEP 4: DISPLAY RESULTS ---
if st.button("🚀 Predict Winner", type="primary"):
    with st.spinner('Simulating 2,000 possible race scenarios...'):
        probs = run_simulation(v1, v2, v3, visible_track, visible_lane)
    
    # Find the most likely winner
    likely_winner = max(probs, key=probs.get)
    st.success(f"🏆 Most Likely Winner: **{likely_winner}**")

    # Display Progress Bars
    st.subheader("Win Probability")
    for vehicle, percent in probs.items():
        col1, col2 = st.columns([1, 3])
        with col1:
            st.write(f"**{vehicle}**")
        with col2:
            st.progress(int(percent))
            st.caption(f"{percent:.1f}% Chance")

    # Show the Stats Table for reference
    st.divider()
    with st.expander("See Raw Speed Stats for Selected Vehicles"):
        df = pd.DataFrame({
            v1: SPEED_DATA[v1],
            v2: SPEED_DATA[v2],
            v3: SPEED_DATA[v3]
        })
        st.dataframe(df.transpose())
