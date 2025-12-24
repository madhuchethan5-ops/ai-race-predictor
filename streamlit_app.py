import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. DATA & SPEED CONFIG ---
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
st.set_page_config(layout="wide", page_title="AI Race Predictor")

# --- 2. GLOBAL COMPACT CSS (ZERO-SCROLL MODE) ---
st.markdown("""
    <style>
    /* Pull everything up and together */
    .block-container {padding-top: 1rem; padding-bottom: 0rem;}
    [data-testid="stSidebar"] {padding-top: 0rem;}
    [data-testid="stVerticalBlock"] {gap: 0.4rem;}
    
    /* Headers Styling */
    h1 {margin-top: -30px; font-weight: 300; text-align: center; font-size: 2rem;}
    h2, h3, h4 {font-weight: 300; margin-bottom: 0px;}
    
    /* Compact Sidebar Elements */
    div.stSelectbox label, div.stRadio label {font-size: 14px; margin-bottom: -10px;}
    div.stMetric {background-color: #f8f9fb; padding: 5px; border-radius: 5px; border: 1px solid #eee;}
    
    /* Form Padding */
    .stForm {padding: 10px; border-radius: 10px;}
    </style>
    """, unsafe_allow_html=True)

# --- 3. SIMULATION ENGINE ---
def run_simulation(v1, v2, v3, visible_t, visible_l):
    wins = {v1: 0, v2: 0, v3: 0}
    iterations = 2000
    all_terrains = list(SPEED_DATA["Car"].keys())
    
    # Learn average visible length from history
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
        
        # Calculate travel time per lane
        times = {v: sum([(lengths[i]/SPEED_DATA[v][t_list[i]]) for i in range(3)]) for v in [v1, v2, v3]}
        wins[min(times, key=times.get)] += 1
    return {k: (v / iterations) * 100 for k, v in wins.items()}

# --- 4. SIDEBAR (ZERO-SCROLL SELECTION) ---
with st.sidebar:
    # PRIMARY BUTTON AT TOP
    predict_clicked = st.button("🚀 RUN AI PREDICTION", type="primary", use_container_width=True)
    
    st.markdown("### 🚦 Race Setup")
    visible_track = st.selectbox("Visible Track", list(SPEED_DATA["Car"].keys()))
    visible_lane = st.radio("Visible Lane", [1, 2, 3], horizontal=True)
    
    st.divider()
    
    st.markdown("### 🏎️ Competitors")
    v1 = st.selectbox("Top Lane", list(SPEED_DATA.keys()), index=8)
    v2 = st.selectbox("Mid Lane", list(SPEED_DATA.keys()), index=7)
    v3 = st.selectbox("Bot Lane", list(SPEED_DATA.keys()), index=5)

# --- 5. MAIN SCREEN RESULTS ---
st.markdown("<h1>AI Race Predictor</h1>", unsafe_allow_html=True)

if predict_clicked:
    probs = run_simulation(v1, v2, v3, visible_track, visible_lane)
    st.session_state['last_pred'] = max(probs, key=probs.get)
    
    st.markdown("#### 🏁 Prediction Results")
    col_win, col_risk = st.columns([2, 1])
    with col_win:
        st.markdown(f"**Pick:** <span style='color: #FF4B4B; font-size: 22px;'>{st.session_state['last_pred']}</span>", unsafe_allow_html=True)
    
    with col_risk:
        # Risk assessment based on probability gap
        sorted_p = sorted(probs.values(), reverse=True)
        gap = sorted_p[0] - sorted_p[1]
        if gap > 40: st.success("✅ LOW RISK")
        elif gap > 15: st.warning("⚠️ MED RISK")
        else: st.error("🚨 HIGH RISK")

    c1, c2, c3 = st.columns(3)
    p_items = list(probs.items())
    for i, col in enumerate([c1, c2, c3]):
        with col:
            st.metric(p_items[i][0], f"{p_items[i][1]:.1f}%")
            st.progress(int(p_items[i][1]))
    st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)

# --- 6. COMPACT LOGGER ---
st.markdown("#### 📝 Race Logger")
with st.form("race_logger", clear_on_submit=True):
    lcol1, lcol2, lcol3 = st.columns(3)
    with lcol1:
        actual_winner = st.selectbox("Actual Winner", [v1, v2, v3])
        vis_len_actual = st.number_input("Visible %", 5, 95, 30)
    with lcol2:
        h1_track = st.selectbox("Hidden 1", list(SPEED_DATA["Car"].keys()))
        h1_len = st.number_input("H1 Len %", 5, 95, 35)
    with lcol3:
        h2_track = st.selectbox("Hidden 2", list(SPEED_DATA["Car"].keys()))
        h2_len = st.number_input("H2 Len %", 5, 95, 35)
    
    if st.form_submit_button("💾 Save to History", use_container_width=True):
        prediction = st.session_state.get('last_pred', "N/A")
        new_row = pd.DataFrame([{
            "V1": v1, "V2": v2, "V3": v3, "Actual_Winner": actual_winner,
            "Lane": visible_lane, "Visible_Track": visible_track,
            "Visible_Lane_Length (%)": vis_len_actual,
            "Hidden_1": h1_track, "Hidden_1_Len": h1_len,
            "Hidden_2": h2_track, "Hidden_2_Len": h2_len, "Predicted_Winner": prediction
        }])
        new_row.to_csv(CSV_FILE, mode='a', header=not os.path.exists(CSV_FILE), index=False)
        st.toast("Race Saved!", icon="✅")

# --- 7. ANALYTICS & DOWNLOAD ---
if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
    st.divider()
    
    # Direct Download
    st.download_button(
        label=f"📥 Download All {len(df)} History Rows (CSV)",
        data=df.to_csv(index=False),
        file_name="race_history_COMPLETE.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.markdown("### 📊 Accuracy & Trends")
    
    # Model Success Rate Tracking
    if 'Actual_Winner' in df.columns and 'Predicted_Winner' in df.columns:
        valid_df = df[df['Predicted_Winner'] != "N/A"].copy()
        if not valid_df.empty:
            valid_df['Is_Correct'] = valid_df['Actual_Winner'] == valid_df['Predicted_Winner']
            st.metric("Model Prediction Success", f"{(valid_df['Is_Correct'].mean()*100):.1f}%")
            
            # Difficulty Table
            st.markdown("#### 🚩 Track Failure Rates")
            diff = valid_df.groupby('Visible_Track')['Is_Correct'].agg(['count', 'mean'])
            diff.columns = ['Races', 'Success %']
            diff['Failure %'] = 100 - (diff['Success %'] * 100)
            st.table(diff[['Races', 'Failure %']].sort_values('Failure %', ascending=False).style.background_gradient(cmap='Reds'))

    # Win Rates Chart
    st.markdown("#### 🏎️ Vehicle Win Counts")
    win_counts = df['Actual_Winner'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.barplot(x=win_counts.index, y=win_counts.values, palette="magma", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    with st.expander("🔍 Full Data View"):
        st.dataframe(df, use_container_width=True)
