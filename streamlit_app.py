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

# Set Page Config for wide layout to use screen space efficiently
st.set_page_config(layout="wide", page_title="AI Race Predictor")

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

# --- 3. SIDEBAR (SPEED-OPTIMIZED) ---
with st.sidebar:
    # Action Button at the TOP for zero-scroll access
    predict_clicked = st.button("🚀 RUN AI PREDICTION", type="primary", use_container_width=True)
    
    st.markdown("<h3 style='font-weight: 300; margin-top: -10px;'>🚦 Setup</h3>", unsafe_allow_html=True)
    
    # Using small labels and compact widgets
    with st.expander("🌍 Environment", expanded=True):
        visible_track = st.selectbox("Track", list(SPEED_DATA["Car"].keys()), label_visibility="collapsed")
        visible_lane = st.radio("Lane", [1, 2, 3], horizontal=True)
    
    st.markdown("<p style='margin-bottom: -15px; font-size: 14px;'>🏎️ <b>Competitors</b></p>", unsafe_allow_html=True)
    # Removing labels to save vertical space
    v1 = st.selectbox("V1", list(SPEED_DATA.keys()), index=8, label_visibility="collapsed")
    v2 = st.selectbox("V2", list(SPEED_DATA.keys()), index=7, label_visibility="collapsed")
    v3 = st.selectbox("V3", list(SPEED_DATA.keys()), index=5, label_visibility="collapsed")
    
    st.divider()
    # Duplicate button at the bottom just in case, but the top one is your primary
    st.button("🚀 Run AI Prediction", key="bottom_btn", use_container_width=True)

# --- 4. MAIN SCREEN (COMPACT LAYOUT & CSS) ---
st.markdown("""
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 0rem;}
    h1 {margin-top: -30px; font-weight: 300; text-align: center; margin-bottom: 0px;}
    .stMetric {background-color: #f8f9fb; padding: 10px; border-radius: 8px; border: 1px solid #e6e9ef;}
    hr {margin: 15px 0;}
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1>AI Race Predictor</h1>", unsafe_allow_html=True)

if predict_clicked:
    probs = run_simulation(v1, v2, v3, visible_track, visible_lane)
    st.session_state['last_pred'] = max(probs, key=probs.get)
    
    st.markdown("<h4 style='font-weight: 300; margin-top: 10px;'>🏁 Prediction Results</h4>", unsafe_allow_html=True)
    
    col_win, col_risk = st.columns([2, 1])
    with col_win:
        st.markdown(f"**Best Strategic Pick:** <span style='color: #FF4B4B; font-size: 20px;'>{st.session_state['last_pred']}</span>", unsafe_allow_html=True)
    
    with col_risk:
        sorted_p = sorted(probs.values(), reverse=True)
        gap = sorted_p[0] - sorted_p[1]
        if gap > 40: st.success("✅ LOW RISK")
        elif gap > 15: st.warning("⚠️ MEDIUM RISK")
        else: st.error("🚨 HIGH RISK")

    c1, c2, c3 = st.columns(3)
    p_items = list(probs.items())
    for i, col in enumerate([c1, c2, c3]):
        with col:
            st.metric(p_items[i][0], f"{p_items[i][1]:.1f}%")
            st.progress(int(p_items[i][1]))
    st.divider()

# --- 5. DATA LOGGING FORM (SINGLE, COMPACT ROW) ---
st.markdown("<h4 style='font-weight: 300;'>📝 Race Outcome Logger</h4>", unsafe_allow_html=True)
with st.form("race_logger", clear_on_submit=True):
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        actual_winner = st.selectbox("Actual Winner", [v1, v2, v3])
        vis_len_actual = st.number_input("Visible Segment %", 5, 95, 30)
    with f_col2:
        h1_track = st.selectbox("Hidden Track 1", list(SPEED_DATA["Car"].keys()))
        h1_len = st.number_input("Hidden 1 %", 5, 95, 35)
    with f_col3:
        h2_track = st.selectbox("Hidden Track 2", list(SPEED_DATA["Car"].keys()))
        h2_len = st.number_input("Hidden 2 %", 5, 95, 35)
    
    if st.form_submit_button("💾 Save Race to History", use_container_width=True):
        prediction = st.session_state.get('last_pred', "N/A")
        new_row = pd.DataFrame([{
            "V1": v1, "V2": v2, "V3": v3,
            "Actual_Winner": actual_winner,
            "Lane": visible_lane,
            "Visible_Track": visible_track,
            "Visible_Lane_Length (%)": vis_len_actual,
            "Hidden_1": h1_track, "Hidden_1_Len": h1_len,
            "Hidden_2": h2_track, "Hidden_2_Len": h2_len,
            "Predicted_Winner": prediction
        }])
        new_row.to_csv(CSV_FILE, mode='a', header=not os.path.exists(CSV_FILE), index=False)
        st.toast("Race saved successfully!", icon="✅")

# --- 6. ANALYTICS & DOWNLOAD (RESCUE MODE) ---
if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
    st.divider()
    
    # 6a. Download History (Always available for all rows)
    st.markdown("<h4 style='font-weight: 300;'>💾 Data Export</h4>", unsafe_allow_html=True)
    st.download_button(
        label=f"📥 Download History ({len(df)} Races)",
        data=df.to_csv(index=False),
        file_name="complete_race_history.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.markdown("<h2 style='font-weight: 300;'>📊 Performance Dashboard</h2>", unsafe_allow_html=True)
    
    # Accuracy & Track Difficulty Table (Calculated only for non-N/A rows)
    if 'Actual_Winner' in df.columns and 'Predicted_Winner' in df.columns:
        valid_df = df[df['Predicted_Winner'] != "N/A"].copy()
        if not valid_df.empty:
            valid_df['Is_Correct'] = valid_df['Actual_Winner'] == valid_df['Predicted_Winner']
            
            col_acc, col_diff = st.columns([1, 2])
            with col_acc:
                acc = valid_df['Is_Correct'].mean() * 100
                st.metric("Overall AI Accuracy", f"{acc:.1f}%")
            
            with col_diff:
                st.markdown("<h4 style='font-weight: 300;'>🚩 Track Failure Rates</h4>", unsafe_allow_html=True)
                diff_df = valid_df.groupby('Visible_Track')['Is_Correct'].agg(['count', 'mean'])
                diff_df.columns = ['Races', 'Success Rate %']
                diff_df['Failure Rate %'] = 100 - (diff_df['Success Rate %'] * 100)
                st.table(diff_df[['Races', 'Failure Rate %']].sort_values('Failure Rate %', ascending=False).style.background_gradient(cmap='Reds'))

    # Win Rate Chart
    st.markdown("<h4 style='font-weight: 300;'>🏎️ Vehicle Win Distribution</h4>", unsafe_allow_html=True)
    win_counts = df['Actual_Winner'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.barplot(x=win_counts.index, y=win_counts.values, palette="magma", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    with st.expander("🔍 View Raw CSV Data"):
        st.dataframe(df, use_container_width=True)
