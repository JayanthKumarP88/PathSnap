import os
import streamlit as st
import requests
import json
import streamlit.components.v1 as components

# Set page config
st.set_page_config(
    page_title="PathSnap | Evaluator Route Reconstruction",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, professional UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0D47A1;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #546E7A;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F8F9FA;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #78909C;
        text-transform: uppercase;
        font-weight: 600;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1A237E;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Target_Corporation_logo_%28vector%29.svg/200px-Target_Corporation_logo_%28vector%29.svg.png", width=60)
st.sidebar.title("⚙️ System Config")

# Auto-probe active Laptop 1 API IP (eliminates manual typing)
candidate_urls = [
    os.environ.get("LAPTOP1_API_URL"),
    "http://10.30.204.40:8000",
    "http://10.2.0.2:8000",
    "http://localhost:8000"
]
default_api_url = "http://10.30.204.40:8000"
for cand in candidate_urls:
    if cand:
        try:
            r = requests.get(f"{cand}/", timeout=0.8)
            if r.status_code == 200:
                default_api_url = cand
                break
        except Exception:
            pass

# Laptop 1 Connection settings with auto-connected URL
api_host = st.sidebar.text_input("Laptop 1 API Base URL", value=default_api_url, help="IP address of Laptop 1")

# Test Connection
try:
    health_resp = requests.get(f"{api_host}/", timeout=2)
    if health_resp.status_code == 200:
        health_info = health_resp.json()
        st.sidebar.success(f"🟢 Connected to Laptop 1\nActive Model: {health_info.get('active_model', 'N/A')}")
    else:
        st.sidebar.error("🔴 Connected, but server reported an error.")
except Exception:
    st.sidebar.warning("⚠️ Laptop 1 API not reachable. Ensure run_api.bat is running on Laptop 1.")

# Fetch available evaluation trips
available_trips = []
try:
    trips_resp = requests.get(f"{api_host}/trips", timeout=2)
    if trips_resp.status_code == 200:
        available_trips = trips_resp.json().get("available_trips", [])
except Exception:
    pass

if not available_trips:
    # Fallback to standard 10 evaluation trips
    available_trips = [
        "00000000", "00000007", "00000009", "00000040", "00000052",
        "00000056", "00000065", "00000076", "00000077", "00000091"
    ]

# Trip Selector
st.sidebar.markdown("---")
st.sidebar.subheader("📍 Select Evaluation Trip")
selected_trip = st.sidebar.selectbox("Reserved Evaluation Trips", available_trips)
use_hmm = st.sidebar.checkbox("Apply HMM Viterbi Decoding", value=True, help="Enforce road network topological connectivity using HMM")
search_radius = st.sidebar.slider("Candidate Search Radius (m)", min_value=20, max_value=60, value=35, step=5)

# Main Dashboard View
st.markdown("<div class='main-header'>PathSnap: Distributed Map-Matching & Route Reconstruction</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>PES University UE24CS352A | Evaluator Control Interface (Laptop 2)</div>", unsafe_allow_html=True)

col_run, col_info = st.columns([1, 4])
with col_run:
    evaluate_clicked = st.button("🚀 Run Reconstruction & Evaluate", type="primary", use_container_width=True)

if evaluate_clicked:
    with st.spinner(f"Sending evaluation request for Trip #{selected_trip} to Laptop 1..."):
        try:
            payload = {
                "trip_id": selected_trip,
                "use_hmm": use_hmm,
                "search_radius": float(search_radius)
            }
            resp = requests.post(f"{api_host}/evaluate", json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()

                st.success(f"Trip #{selected_trip} evaluated successfully in {data['inference_time_sec']} seconds!")

                # Display KPI Cards
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    acc_val = data['top1_accuracy']
                    acc_display = f"{acc_val * 100:.2f}%" if isinstance(acc_val, (int, float)) else str(acc_val)
                    st.metric("Top-1 Accuracy", acc_display)
                with col2:
                    st.metric("Route Continuity", f"{data['route_continuity'] * 100:.2f}%")
                with col3:
                    st.metric("Candidate Miss Rate", f"{data['candidate_miss_rate'] * 100:.2f}%")
                with col4:
                    st.metric("Total Observations", f"{data['observations_count']}")
                with col5:
                    st.metric("Inference Latency", f"{data['inference_time_sec']} s")

                # Fetch and render Folium Map from Laptop 1
                st.markdown("---")
                st.subheader(f"🗺️ Reconstructed Route Visualization (Trip {selected_trip})")
                map_url = f"{api_host}/map/{selected_trip}?use_hmm={str(use_hmm).lower()}"
                map_resp = requests.get(map_url, timeout=30)
                if map_resp.status_code == 200:
                    components.html(map_resp.text, height=580, scrolling=False)
                else:
                    st.error("Could not render Folium map from API.")

                # Reconstructed sequence preview
                st.markdown("---")
                st.subheader("📋 Chronological Reconstructed Arc Sequence (First 50 Segments)")
                rec_arcs = data.get("reconstructed_arcs", [])
                st.write(rec_arcs[:50])

            else:
                st.error(f"Server error: {resp.status_code} - {resp.text}")
        except Exception as ex:
            st.error(f"Failed to communicate with Laptop 1 API: {str(ex)}")
else:
    st.info("👈 Select any of the ten reserved evaluation trips from the sidebar and click **'Run Reconstruction & Evaluate'**.")
    st.markdown("""
    ### Evaluation Guidelines
    1. **Laptop 1 API**: Ensure the FastAPI server is running (`run_api.bat`).
    2. **Ground Truth Integrity**: All true route labels remain strictly on Laptop 1 and are evaluated server-side.
    3. **Live Logs**: Watch the terminal on Laptop 1 to verify real-time inference requests and processing timestamps.
    """)
