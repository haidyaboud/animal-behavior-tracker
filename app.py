import streamlit as st
import tempfile
import base64
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from object_detection import run_object_detection  # Custom module

# --- Page Config ---
st.set_page_config(
    page_title="Animal Behavior Tracker",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
    <style>
    .main { background: #f8f9fa; font-family: 'Arial', sans-serif; padding: 2rem; }
    .hero-section {
        background: linear-gradient(135deg, #1a2e44, #2e4a66);
        color: white;
        padding: 4rem 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 3rem;
    }
    .hero-section h1 { font-size: 3rem; margin-bottom: 1rem; }
    .hero-section p { font-size: 1.25rem; margin-bottom: 2rem; }
    .cta-button {
        background: white; color: #1a2e44;
        padding: 0.8rem 2rem; font-size: 1.1rem;
        border-radius: 25px; text-decoration: none;
        transition: background 0.3s ease;
    }
    .cta-button:hover { background: #e0e0e0; }
    .card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        text-align: center;
        max-width: 1000px;
        margin: 0 auto 2rem auto;
    }
    .card h3 { color: #1a2e44; margin-bottom: 1rem; font-size: 1.75rem; }
    .card p { color: #5e6c84; font-size: 1.1rem; margin-bottom: 1rem; }
    .icon { font-size: 2.5rem; margin-bottom: 1rem; }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
st.sidebar.title("🔍 Navigation")
if st.sidebar.button("🏠 Home"):
    st.session_state.page = "home"
    st.rerun()
if st.sidebar.button("🎯 Tracking"):
    st.session_state.page = "tracking"
    st.rerun()
if "page" not in st.session_state:
    st.session_state.page = "home"

# --- Helper: File Download ---
def get_binary_file_downloader_html(file_path, file_label):
    with open(file_path, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    return f'<a href="data:application/octet-stream;base64,{bin_str}" download="{file_label}">📥 Download {file_label}</a>'

# --- HOME PAGE ---
def home_page():
    st.markdown("""
        <div class="hero-section">
            <h1>Animal Behavior Tracker</h1>
            <p>Explore and analyze animal behaviors with advanced AI-powered video tracking technology.</p>
            <a href="?page=tracking" class="cta-button">🚀 Start Tracking Now</a>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="card">
            <div class="icon">🔍</div>
            <h3>Advanced Object Detection</h3>
            <p>Track animals in videos with precision using our AI-powered object detection.</p>
            <a href="?page=tracking" class="cta-button">Try Object Detection</a>
        </div>
    """, unsafe_allow_html=True)

    # --- Trajectory per Cluster ---
    st.markdown("""
        <div class="card">
            <h3>1️⃣ Representative Trajectory per Cluster</h3>
            <p>Sample trajectories of animals grouped by cluster with start/end markers.</p>
        </div>
    """, unsafe_allow_html=True)
    cluster_data = pd.DataFrame({
        "X": [960, 980, 1000, 760, 765, 770, 730, 740, 750],
        "Y": [905, 910, 880, 810, 800, 782, 785, 800, 828],
        "Cluster": ["Cluster 0"] * 3 + ["Cluster 1"] * 3 + ["Cluster 2"] * 3
    })
    fig1 = px.line(cluster_data, x="X", y="Y", color="Cluster", markers=True,
                   title="Representative Trajectory per Cluster",
                   color_discrete_map={"Cluster 0": "red", "Cluster 1": "green", "Cluster 2": "blue"})
    fig1.update_layout(xaxis_title="X Coordinate", yaxis_title="Y Coordinate")
    st.plotly_chart(fig1, use_container_width=True)

    # --- Number of Segments per Cluster ---
    st.markdown("""
        <div class="card">
            <h3>2️⃣ Number of Trajectory Segments per Cluster</h3>
            <p>How frequently each behavior cluster appears in the video.</p>
        </div>
    """, unsafe_allow_html=True)
    segment_df = pd.DataFrame({
        "Cluster": ["Cluster 0", "Cluster 1", "Cluster 2"],
        "Segments": [93, 51, 168],
        "Percentage": ["29.8%", "16.3%", "53.8%"]
    })
    fig2 = px.bar(segment_df, x="Cluster", y="Segments", text="Percentage",
                  color="Cluster", color_discrete_map={
                      "Cluster 0": "red", "Cluster 1": "green", "Cluster 2": "blue"
                  },
                  title="Number of Trajectory Segments per Cluster")
    fig2.update_traces(textposition="outside")
    fig2.update_layout(yaxis_title="Number of Segments")
    st.plotly_chart(fig2, use_container_width=True)

    # --- Temporal Distribution ---
    st.markdown("""
        <div class="card">
            <h3>3️⃣ Temporal Distribution of Clusters</h3>
            <p>Shows the distribution of segments across the timeline of the video.</p>
        </div>
    """, unsafe_allow_html=True)
    time = np.arange(0, 900, 50)
    df_temporal = pd.DataFrame({
        "Frame": list(time) * 3,
        "Cluster": ["Cluster 0"] * len(time) + ["Cluster 1"] * len(time) + ["Cluster 2"] * len(time),
        "Count": np.random.randint(5, 30, len(time)).tolist() +
                 np.random.randint(3, 18, len(time)).tolist() +
                 np.random.randint(10, 35, len(time)).tolist()
    })
    fig3 = px.line(df_temporal, x="Frame", y="Count", color="Cluster",
                   title="Temporal Distribution of Clusters",
                   color_discrete_map={"Cluster 0": "red", "Cluster 1": "green", "Cluster 2": "blue"})
    fig3.update_layout(xaxis_title="Frame Number", yaxis_title="Number of Segments")
    st.plotly_chart(fig3, use_container_width=True)

# --- TRACKING PAGE ---
def tracking_page():
    st.markdown("<h1 style='text-align:center;color:#1a2e44;'>🐾 Animal Object Tracking</h1>", unsafe_allow_html=True)

    animal_options = {
        "Chicks": {"symbol": "🐥", "description": "Use videos with clear visibility of individual birds."},
        "Rats": {"symbol": "🐀", "description": "Ensure good contrast between rats and background."},
        "Fish": {"symbol": "🐟", "description": "Use videos with minimal reflections."}
    }

    selected_animal = st.radio(
        "👉 Select Animal Type",
        options=list(animal_options.keys()),
        format_func=lambda x: f"{animal_options[x]['symbol']} {x}",
        horizontal=True
    )

    st.info(animal_options[selected_animal]["description"])

    uploaded_file = st.file_uploader("📂 Upload a video file (mp4, avi, mov)", type=["mp4", "avi", "mov"])
    if uploaded_file:
        st.video(uploaded_file)
        with st.expander("⚙️ Advanced Options", expanded=True):
            confidence_threshold = st.slider("🎯 Confidence Threshold", 0.1, 0.9, 0.5, 0.05)

        if st.button(f"🚀 Track {selected_animal}"):
            try:
                with st.spinner("🔍 Processing video..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        video_path = tmp_file.name

                    detection_results = run_object_detection(
                        video_path=video_path,
                        animal_type=selected_animal,
                        confidence_threshold=confidence_threshold
                    )

                st.success("✅ Tracking completed!")
                if detection_results["output_video"]:
                    st.markdown(get_binary_file_downloader_html(detection_results["output_video"], "detected_video.mp4"), unsafe_allow_html=True)
                else:
                    st.error("❌ No output video generated.")
            except Exception as e:
                st.error(f"❌ Error: {e}")
            finally:
                if os.path.exists(video_path):
                    os.remove(video_path)

# --- PAGE ROUTER ---
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "tracking":
    tracking_page()
else:
    st.error("🚫 Page not found!")
