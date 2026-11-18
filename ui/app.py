import os
import streamlit as st
import streamlit.components.v1 as components

# Configure page metadata
st.set_page_config(
    page_title="PathSnap — Distributed GPS Map-Matching",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Seamless edge-to-edge layout styling
st.markdown("""
<style>
    #MainMenu, header, footer, .stDeployButton, [data-testid="stToolbar"], [data-testid="stDecoration"] {
        display: none !important;
        visibility: hidden !important;
    }
    div[data-testid="stSidebarNav"] {display: none !important;}
    section[data-testid="stSidebar"] {display: none !important;}
    .main, .block-container {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100vw !important;
        height: 100vh !important;
        overflow: hidden !important;
    }
    iframe {
        width: 100vw !important;
        height: 100vh !important;
        border: none !important;
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        z-index: 999999 !important;
    }
</style>
""", unsafe_allow_html=True)

# Path to self-contained web app
current_dir = os.path.dirname(os.path.abspath(__file__))
index_path = os.path.join(current_dir, "..", "web", "index.html")

target_api = os.environ.get("LAPTOP1_API_URL", "http://localhost:8000")

with open(index_path, "r", encoding="utf-8") as f:
    html_content = f.read()

# Render native HTML/CSS/JS Leaflet dashboard full screen
components.html(html_content, height=1000, scrolling=False)
