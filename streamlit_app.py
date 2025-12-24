import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

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

# --- 2. SIMULATION ENGINE ---
def run_simulation(v1, v2, v3, visible_t, visible_l):
    wins = {v1: 0, v2: 0, v3: 0}
    iterations = 2000
    all_terrains = list(SPEED_DATA["Car"].keys())
    
    avg_vis_len = 0.30 
    if os.path.exists(CSV_FILE):
        df_h = pd.read_csv(CSV_FILE)
        if 'Visible_Lane_Length (%)' in df_h.columns and not df_h[df_h['Visible_Track'] == visible_t].empty:
            avg_vis_len = df_h[df_h['Visible_Track'] == visible_t]['Visible_Lane_Length (%)'].mean() / 100

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
        
        times = {v: sum([(lengths[i]/SPEED_DATA[v][t_list[i]]) for i in range(3)]) for v in [v1, v2, v3]}
        wins[min(times, key=times.get)] += 1
    return {k: (v / iterations) * 100 for k, v in wins.items()}

# --- 3. SIDEBAR (COMPACT INPUTS) ---
with st.sidebar:
    st.markdown("<h2 style='font-weight: 300;'>🚦 Setup</h2>", unsafe_allow_html=True)
    with st.expander("🌍 Environment", expanded=True):
        visible_track = st.selectbox("Track", list(SPEED_DATA["Car"].keys()))
        visible_lane = st.radio("Lane", [1, 2, 3], horizontal=True)
    
    with st.expander("🏎️ Competitors", expanded=True):
        v1 = st.selectbox("Top Car", list(SPEED_DATA.keys()), index=8)
        v2 = st.selectbox("Mid Car", list(SPEED_DATA.keys()), index=7)
        v3 = st.selectbox("Bot Car", list(SPEED_DATA.keys()), index=5)
    
    predict_clicked = st.button("🚀 Run Prediction", type="primary", use_container_width=True)

# --- 4. MAIN SCREEN (COMPACT LAYOUT) ---
# This CSS pulls the elements closer together and reduces top margins
st.markdown("""
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 0rem;}
    h1 {margin-top: -30px; padding-bottom: 10px;}
    div[data-testid="stExpander"] {margin-top: -15px;}
    .stProgress {margin-top: -20px;}
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h2 style='font-weight: 300; text-align: center; margin-bottom: 0px;'>AI Race Predictor</h2>", unsafe_allow_html=True)

if predict_clicked:
    probs = run_simulation(v1, v2, v3, visible_track, visible_lane)
    st.session_state['last_pred'] = max(probs, key=probs.get)
    
    # Prediction Results Header
    st.markdown("<h4 style='font-weight: 300; margin-top: 10px; margin-bottom: 5px;'>🏁 Results</h4>", unsafe_allow_html=True)
    
    col_win, col_risk = st.columns([2, 1])
    with col_win:
        st.markdown(f"**Top Pick:** <span style='color: #FF4B4B;'>{st.session_state['last_pred']}</span>", unsafe_allow_html=True)
    with col_risk:
        sorted_p = sorted(probs.values(), reverse=True)
        gap = sorted_p[0] - sorted_p[1]
        if gap > 40: st.success("✅ LOW RISK")
        elif gap > 15: st.warning("⚠️ MED RISK")
        else: st.error("🚨 HIGH RISK")

    # Metrics side-by-side with reduced spacing
    c1, c2, c3 = st.columns(3)
    p_items = list(probs.items())
    for i, col in enumerate([c1, c2, c3]):
        with col:
            st.markdown(f"<p style='font-size: 14px; margin-bottom: 0px;'>{p_items[i][0]}</p>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='margin-top: -5px;'>{p_items[i][1]:.1f}%</h3>", unsafe_allow_html=True)
            st.progress(int(p_items[i][1]))
    st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)

# --- 5. LOGGING FORM (COMPACT) ---
st.markdown("<h4 style='font-weight: 300; margin-bottom: 5px;'>📝 Race Logger</h4>", unsafe_allow_html=True)
with st.form("race_logger", clear_on_submit=True):
    lcol1, lcol2, lcol3 = st.columns(3)
    with lcol1:
        actual_winner = st.selectbox("Winner", [v1, v2, v3])
        vis_len_actual = st.number_input("Visible Segment %", 5, 95, 30)
    with lcol2:
        h1_track = st.selectbox("Hidden 1", list(SPEED_DATA["Car"].keys()))
        h1_len = st.number_input("H1 Length %", 5, 95, 35)
    with lcol3:
        h2_track = st.selectbox("Hidden 2", list(SPEED_DATA["Car"].keys()))
        h2_len = st.number_input("H2 Length %", 5, 95, 35)
    
    # Save button
    st.form_submit_button("💾 Save Result", use_container_width=True)

# --- 6. ANALYTICS ---
if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
    st.divider()
    st.markdown("<h2 style='font-weight: 300;'>📊 Performance Analytics</h2>", unsafe_allow_html=True)

    if 'Actual_Winner' in df.columns and 'Predicted_Winner' in df.columns:
        valid_df = df[df['Predicted_Winner'] != "N/A"].copy()
        if not valid_df.empty:
            valid_df['Is_Correct'] = valid_df['Actual_Winner'] == valid_df['Predicted_Winner']
            
            # Accuracy & Difficulty
            st.metric("Model Accuracy", f"{(valid_df['Is_Correct'].mean()*100):.1f}%")
            
            st.markdown("<h4 style='font-weight: 300;'>🚩 Track Failure Rates</h4>", unsafe_allow_html=True)
            diff = valid_df.groupby('Visible_Track')['Is_Correct'].agg(['count', 'mean'])
            diff.columns = ['Races', 'Success %']
            diff['Failure %'] = 100 - (diff['Success %'] * 100)
            st.table(diff[['Races', 'Failure %']].sort_values('Failure %', ascending=False).style.background_gradient(cmap='Reds'))

    # Wins Chart
    st.markdown("<h4 style='font-weight: 300;'>🏎️ Win Distribution</h4>", unsafe_allow_html=True)
    win_counts = df['Actual_Winner'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.barplot(x=win_counts.index, y=win_counts.values, palette="magma", ax=ax)
    st.pyplot(fig)

    # Backup Download
    with open(CSV_FILE, 'rb') as f:
        st.download_button("📥 Download History", f, "race_history.csv", "text/csv", use_container_width=True)
