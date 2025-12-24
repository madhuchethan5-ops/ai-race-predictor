import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import plotly.express as px
from streamlit_extras.stylable_container import stylable_container
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

# --- 2. UI STYLING ---
st.markdown("""
    <style>
    .stApp { background: #0f172a; color: #f8fafc; }
    div[data-testid="stMetric"] { background: rgba(255, 255, 255, 0.05); border-radius: 10px; padding: 15px; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOGIC ---
def load_history():
    if os.path.exists(CSV_FILE): return pd.read_csv(CSV_FILE)
    return pd.DataFrame()

def run_simulation_vectorized(v1, v2, v3, visible_t, visible_l, history_df, iterations=5000):
    vehicles = [v1, v2, v3]
    all_terrains = list(SPEED_DATA["Car"].keys())
    
    # 1. Historical Variance
    avg_vis = 0.30
    if not history_df.empty and 'Visible_Track' in history_df.columns:
        match = history_df[history_df['Visible_Track'] == visible_t]
        if not match.empty: avg_vis = match['Visible_Segment_%'].mean() / 100

    # 2. Vectorized Math
    vis_lens = np.clip(np.random.normal(avg_vis, 0.1, iterations), 0.05, 0.95)
    h1_lens = (1.0 - vis_lens) * np.random.uniform(0.2, 0.8, iterations)
    h2_lens = 1.0 - vis_lens - h1_lens

    seg_terrains = np.random.choice(all_terrains, size=(iterations, 3))
    seg_terrains[:, visible_l-1] = visible_t

    results = {}
    for v in vehicles:
        # Map terrain names to speeds
        speed_map = np.vectorize(SPEED_DATA[v].get)(seg_terrains)
        # 5% Performance Noise
        noise = np.random.normal(1.0, 0.05, (iterations, 3))
        noisy_speeds = speed_map * noise
        times = (vis_lens/noisy_speeds[:, 0]) + (h1_lens/noisy_speeds[:, 1]) + (h2_lens/noisy_speeds[:, 2])
        results[v] = times

    winners = np.argmin(np.array([results[v] for v in vehicles]), axis=0)
    counts = pd.Series(winners).value_counts(normalize=True).sort_index() * 100
    return {vehicles[i]: counts.get(i, 0) for i in range(3)}

# --- 4. INTERFACE ---
history = load_history()

with st.sidebar:
    st.header("🚦 Race Setup")
    v_track = st.selectbox("Visible Track", list(SPEED_DATA["Car"].keys()))
    v_lane = st.radio("Active Lane", [1, 2, 3], horizontal=True)
    c1 = st.selectbox("Vehicle 1", list(SPEED_DATA.keys()), index=8)
    c2 = st.selectbox("Vehicle 2", list(SPEED_DATA.keys()), index=7)
    c3 = st.selectbox("Vehicle 3", list(SPEED_DATA.keys()), index=5)
    predict_btn = st.button("🚀 EXECUTE PREDICTION", use_container_width=True, type="primary")

st.title("🏎️ AI RACE PREDICTOR PRO")

if predict_btn:
    probs = run_simulation_vectorized(c1, c2, c3, v_track, v_lane, history)
    st.session_state['last_probs'] = probs
    
    # Results Grid
    m_grid = grid(3, vertical_align="center")
    for veh, val in probs.items():
        m_grid.metric(veh, f"{val:.1f}%")

    col_chart, col_risk = st.columns([2, 1])
    with col_chart:
        fig = px.pie(names=list(probs.keys()), values=list(probs.values()), hole=0.4, title="Win Probabilities")
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col_risk:
        st.subheader("🚨 Strategic Risk")
        gap = max(probs.values()) - sorted(probs.values())[1]
        if gap > 30: st.success("HIGH CONFIDENCE")
        else: st.error("HIGH VOLATILITY")

# --- 5. LOGGING ---
st.divider()
with st.form("logger", clear_on_submit=True):
    st.subheader("📝 POST-RACE TELEMETRY")
    f1, f2, f3, f4 = st.columns(4)
    with f1: winner = st.selectbox("Actual Winner", [c1, c2, c3])
    with f2: v_len = st.number_input("Visible %", 5, 95, 30)
    with f3: h1_t = st.selectbox("Hidden 1", list(SPEED_DATA["Car"].keys()))
    with f4: h2_t = st.selectbox("Hidden 2", list(SPEED_DATA["Car"].keys()))
    
    if st.form_submit_button("💾 ARCHIVE DATA", use_container_width=True):
        prediction = max(st.session_state.get('last_probs', {}), key=st.session_state.get('last_probs', {}).get, default="N/A")
        new_row = pd.DataFrame([{"Visible_Track": v_track, "Visible_Segment_%": v_len, "Predicted": prediction, "Actual": winner}])
        new_row.to_csv(CSV_FILE, mode='a', header=not os.path.exists(CSV_FILE), index=False)
        st.toast("Telemetry synchronized!")

# --- 6. HISTORY ---
if not history.empty:
    with st.expander("📊 PERFORMANCE DASHBOARD"):
        acc = (history['Predicted'] == history['Actual']).mean() * 100
        st.metric("AI Accuracy", f"{acc:.1f}%")
        st.dataframe(history.tail(5), use_container_width=True)
