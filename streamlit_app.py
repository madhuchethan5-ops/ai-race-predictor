import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
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

# --- 2. ADAPTIVE SIMULATION ENGINE (KEYERROR-PROOF) ---
def run_simulation_vectorized(v1, v2, v3, visible_t, visible_l, history_df, iterations=5000):
    vehicles = [v1, v2, v3]
    all_terrains = list(SPEED_DATA["Car"].keys())
    
    avg_vis = 0.33
    vis_std = 0.08
    
    # Check if the column exists before trying to use it
    if not history_df.empty and 'Visible_Segment_%' in history_df.columns:
        match = history_df[history_df['Visible_Track'] == visible_t].tail(20)
        if not match.empty:
            avg_vis = match['Visible_Segment_%'].mean() / 100
            if len(match) > 1:
                vis_std = max(0.04, match['Visible_Segment_%'].std() / 100)

    # Rest of simulation...
    vis_lens = np.clip(np.random.normal(avg_vis, vis_std, iterations), 0.05, 0.95)
    h1_lens = (1.0 - vis_lens) * np.random.uniform(0.1, 0.9, iterations)
    h2_lens = 1.0 - vis_lens - h1_lens

    seg_terrains = np.random.choice(all_terrains, size=(iterations, 3))
    seg_terrains[:, visible_l-1] = visible_t

    results = {}
    for v in vehicles:
        speed_lookup = np.vectorize(SPEED_DATA[v].get)(seg_terrains)
        noise = np.random.normal(1.0, 0.02, (iterations, 3)) # 2% performance variance
        noisy_speeds = speed_lookup * noise
        times = (vis_lens/noisy_speeds[:, 0]) + (h1_lens/noisy_speeds[:, 1]) + (h2_lens/noisy_speeds[:, 2])
        results[v] = times

    winners = np.argmin(np.array([results[v] for v in vehicles]), axis=0)
    counts = pd.Series(winners).value_counts(normalize=True).sort_index() * 100
    return {vehicles[i]: counts.get(i, 0) for i in range(3)}

# --- 3. DATA LOADING (STABILIZED) ---
if os.path.exists(CSV_FILE):
    history = pd.read_csv(CSV_FILE)
    # SANITIZER: Rename old column names to new standard immediately upon loading
    rename_map = {
        'Predicted_Winner': 'Predicted',
        'Actual_Winner': 'Actual',
        'Visible_Lane_Length (%)': 'Visible_Segment_%',
        'Visible_%': 'Visible_Segment_%'
    }
    history = history.rename(columns=rename_map)
else:
    history = pd.DataFrame()

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
    # Pass 'history' (defined above) to the function
    probs = run_simulation_vectorized(c1, c2, c3, v_track, v_lane, history)
    st.session_state['last_probs'] = probs
    st.session_state['last_vehicles'] = [c1, c2, c3]
    
    m_grid = grid(3, vertical_align="center")
    for veh, val in probs.items():
        m_grid.metric(veh, f"{val:.1f}%")

    col_chart, col_risk = st.columns([2, 1])
    with col_chart:
        fig = px.pie(names=list(probs.keys()), values=list(probs.values()), 
                     hole=0.4, title="Probability of Winning",
                     color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col_risk:
        st.subheader("🚨 Strategic Risk")
        gap = max(probs.values()) - sorted(probs.values())[-2]
        if gap > 30: st.success("HIGH CONFIDENCE")
        else: st.warning("VOLATILE RACE")

# --- 6. DETAILED TELEMETRY ---
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

    submitted = st.form_submit_button("💾 SYNC TELEMETRY TO AI BRAIN", use_container_width=True)
    
    if submitted:
        last_probs = st.session_state.get('last_probs', {})
        predicted = max(last_probs, key=last_probs.get) if last_probs else "N/A"
        
        log_entry = {
            "Visible_Track": v_track,
            "Visible_Segment_%": v_len,
            "Hidden_1_Track": h1_t,
            "Hidden_1_Len": h1_l,
            "Hidden_2_Track": h2_t,
            "Hidden_2_Len": h2_l,
            "Predicted": predicted,
            "Actual": winner
        }
        
        pd.DataFrame([log_entry]).to_csv(CSV_FILE, mode='a', header=not os.path.exists(CSV_FILE), index=False)
        st.toast("Telemetry data synchronized!", icon="⚡")
        st.rerun()

# --- 7. ANALYTICS & ACCURACY HEATMAP (STABILIZED) ---
if not history.empty:
    st.divider()
    st.header("📈 AI Learning Analytics")
    
    # --- AUTO-FIX COLUMN NAMES ---
    # This block prevents the KeyError by mapping old names to the new logic
    rename_map = {
        'Predicted_Winner': 'Predicted',
        'Actual_Winner': 'Actual',
        'Visible_Lane_Length (%)': 'Visible_Segment_%',
        'Visible_%': 'Visible_Segment_%'
    }
    history = history.rename(columns=rename_map)

    # Ensure the columns actually exist before filtering
    if 'Predicted' in history.columns and 'Actual' in history.columns:
        valid_history = history[(history['Predicted'] != "N/A") & (history['Actual'].notna())].copy()
        
        if not valid_history.empty:
            valid_history['Is_Correct'] = (valid_history['Predicted'] == valid_history['Actual']).astype(int)
            
            # 7a. Learning Progress Metrics
            col_acc, col_sample = st.columns(2)
            with col_acc:
                global_acc = valid_history['Is_Correct'].mean() * 100
                st.metric("Global AI Accuracy", f"{global_acc:.1f}%")
            
            with col_sample:
                # Use the first 20 races as the baseline as requested
                st.metric("Total Races Learned", f"{len(valid_history)}", 
                          delta="Baseline: 20 Races" if len(valid_history) >= 20 else None)

            # 7b. Accuracy Heatmap
            st.subheader("🎯 Track-Specific Accuracy Heatmap")
            if 'Visible_Track' in valid_history.columns:
                heatmap_data = valid_history.groupby('Visible_Track')['Is_Correct'].agg(['count', 'mean'])
                heatmap_data.columns = ['Races Observed', 'Accuracy (%)']
                heatmap_data['Accuracy (%)'] = heatmap_data['Accuracy (%)'] * 100
                
                st.dataframe(
                    heatmap_data.sort_values(by='Accuracy (%)', ascending=False)
                    .style.background_gradient(cmap='RdYlGn', subset=['Accuracy (%)'])
                    .format("{:.1f}%", subset=['Accuracy (%)']),
                    use_container_width=True
                )
            
            # 7c. Learning Curve (Rolling Accuracy)
            if len(valid_history) > 3:
                st.subheader("📉 AI Learning Curve")
                # Calculate rolling accuracy over the last 20 races
                valid_history['Rolling_Acc'] = valid_history['Is_Correct'].rolling(window=min(20, len(valid_history))).mean() * 100
                
                fig_trend = px.line(valid_history, 
                                    y='Rolling_Acc', 
                                    title="Success Rate Over Time (20-Race Window)",
                                    labels={'Rolling_Acc': 'Accuracy %', 'index': 'Race Order'})
                fig_trend.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_trend, use_container_width=True)

    with st.expander("🔍 View Full Telemetry Log"):
        # Sort so the newest races are at the top
        st.dataframe(history.sort_index(ascending=False), use_container_width=True)
        
        # Download button for backup
        csv_data = history.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Backup History (CSV)", data=csv_data, file_name="race_history_backup.csv", mime="text/csv")
