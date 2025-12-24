import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- DATA ---
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

# --- SIMULATION ENGINE ---
def run_simulation(v1, v2, v3, visible_t, visible_l):
    wins = {v1: 0, v2: 0, v3: 0}
    iterations = 2000
    all_terrains = list(SPEED_DATA["Car"].keys())
    
    # Try to learn average length from history
    avg_vis_len = 0.30 
    if os.path.exists(CSV_FILE):
        df_h = pd.read_csv(CSV_FILE)
        col_name = 'Visible_Lane_Length (%)'
        if col_name in df_h.columns and not df_h[df_h['Visible_Track'] == visible_t].empty:
            avg_vis_len = df_h[df_h['Visible_Track'] == visible_t][col_name].mean() / 100

    for _ in range(iterations):
        vis_len = np.clip(np.random.normal(avg_vis_len, 0.1), 0.05, 0.95)
        rem_len = 1.0 - vis_len
        h1_len = rem_len * np.random.uniform(0.2, 0.8)
        h2_len = rem_len - h1_len
        
        lengths = [0, 0, 0]
        lengths[visible_l-1] = vis_len
        others = [i for i in range(3) if i != visible_l-1]
        lengths[others[0]] = h1_len
        lengths[others[1]] = h2_len

        t_list = [None, None, None]
        t_list[visible_l-1] = visible_t
        t_list[others[0]] = np.random.choice(all_terrains)
        t_list[others[1]] = np.random.choice(all_terrains)
        
        times = {}
        for v in [v1, v2, v3]:
            times[v] = sum([(lengths[i]/SPEED_DATA[v][t_list[i]]) for i in range(3)])
        
        winner = min(times, key=times.get)
        wins[winner] += 1
    return {k: (v / iterations) * 100 for k, v in wins.items()}

# --- UI ---
st.title("🏎️ AI Race Strategic Predictor")

with st.sidebar:
    st.header("Race Setup")
    v1 = st.selectbox("Vehicle 1", list(SPEED_DATA.keys()), index=8)
    v2 = st.selectbox("Vehicle 2", list(SPEED_DATA.keys()), index=7)
    v3 = st.selectbox("Vehicle 3", list(SPEED_DATA.keys()), index=5)
    visible_track = st.selectbox("Visible Track", list(SPEED_DATA["Car"].keys()))
    visible_lane = st.radio("Visible Lane", [1, 2, 3])

if st.button("🚀 Predict Before Start", type="primary"):
    probs = run_simulation(v1, v2, v3, visible_track, visible_lane)
    st.session_state['last_pred'] = max(probs, key=probs.get)
    st.success(f"🏆 Best Strategic Pick: {st.session_state['last_pred']}")
    for v, p in probs.items():
        st.write(f"**{v}**: {p:.1f}%")
        st.progress(int(p))
# --- STEP 4: DATA LOGGING FORM ---
st.divider()
st.header("📝 Log Race Results")
st.info("Fill this out after the race ends to train your AI.")

with st.form("race_logger"):
    col1, col2, col3 = st.columns(3)
    with col1:
        actual_winner = st.selectbox("Who Actually Won?", [v1, v2, v3])
        vis_len_actual = st.number_input("Visible Lane %", 5, 95, 30)
    with col2:
        h1_track = st.selectbox("Hidden Track 1", list(SPEED_DATA["Car"].keys()))
        h1_len = st.number_input("Hidden 1 %", 5, 95, 35)
    with col3:
        h2_track = st.selectbox("Hidden Track 2", list(SPEED_DATA["Car"].keys()))
        h2_len = st.number_input("Hidden 2 %", 5, 95, 35)
    
    submitted = st.form_submit_button("💾 Save Race to History")

if submitted:
    # Get the prediction from the AI (stored in memory)
    prediction = st.session_state.get('last_pred', "N/A")
    
    # Create the new row
    new_row = pd.DataFrame([{
        "V1": v1, "V2": v2, "V3": v3,
        "Actual_Winner": actual_winner,
        "Lane": visible_lane,
        "Visible_Track": visible_track,
        "Visible_Lane_Length (%)": vis_len_actual,
        "Hidden_1": h1_track,
        "Hidden_1_Len": h1_len,
        "Hidden_2": h2_track,
        "Hidden_2_Len": h2_len,
        "Predicted_Winner": prediction
    }])
    
    # Append to the CSV
    new_row.to_csv(CSV_FILE, mode='a', header=not os.path.exists(CSV_FILE), index=False)
    st.success("✅ Race saved! Refresh the page to see updated Analytics.")

# --- ANALYTICS ---
st.divider()
if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
    st.subheader("📊 Performance Analytics")
    
    # 1. Accuracy Check
    if 'Actual_Winner' in df.columns and 'Predicted_Winner' in df.columns:
        valid_df = df.dropna(subset=['Actual_Winner', 'Predicted_Winner'])
        if not valid_df.empty:
            acc = (len(valid_df[valid_df['Actual_Winner'] == valid_df['Predicted_Winner']]) / len(valid_df)) * 100
            st.metric("AI Accuracy", f"{acc:.1f}%")
    
    # 2. Heatmap Check
    req_cols = ['Visible_Track', 'Visible_Lane_Length (%)', 'Hidden_1', 'Hidden_1_Len', 'Hidden_2', 'Hidden_2_Len']
    if all(c in df.columns for c in req_cols):
        st.subheader("🗺️ Track Length Heatmap")
        t_list = []
        for _, r in df.iterrows():
            t_list.append({'T': r['Visible_Track'], 'L': r['Visible_Lane_Length (%)']})
            t_list.append({'T': r['Hidden_1'], 'L': r['Hidden_1_Len']})
            t_list.append({'T': r['Hidden_2'], 'L': r['Hidden_2_Len']})
        
        plot_df = pd.DataFrame(t_list)
        fig, ax = plt.subplots()
        sns.barplot(data=plot_df, x='T', y='L', ax=ax, palette="magma")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.warning("Heatmap requires detailed columns: Visible_Lane_Length (%), Hidden_1, etc.")

    with st.expander("View Raw Data"):
        st.dataframe(df)
