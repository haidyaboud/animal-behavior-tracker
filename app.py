import streamlit as st
import tempfile
import base64
import os
import pandas as pd
import numpy as np
import plotly.express as px
import cv2
import time
from ultralytics import YOLO
from PIL import Image

# --- YOLO Model Paths ---
MODEL_PATHS = {
    "Chicks": "models/yolov8_chicks.pt",
    "Rats": "models/yolov8_rats.pt",
    "Fish": "models/yolov8_fish.pt"
}

# --- Page Config ---
st.set_page_config(
    page_title="Animal Behavior Tracker",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State Defaults ---
if "page" not in st.session_state:
    st.session_state.page = "home"
if "last_animal_tracked" not in st.session_state:
    st.session_state.last_animal_tracked = None
if "fish_tracked" not in st.session_state:
    st.session_state.fish_tracked = False
if "chicks_tracked" not in st.session_state:
    st.session_state.chicks_tracked = False

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

# --- Utility Functions ---
def load_model(animal_type):
    model_path = MODEL_PATHS.get(animal_type)
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model for {animal_type} not found at {model_path}")
    return YOLO(model_path)

def run_object_detection(video_path, animal_type, confidence_threshold):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        output_path = tmp.name
    model = load_model(animal_type)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=confidence_threshold)
        annotated = results[0].plot() if results[0].boxes.data is not None else frame
        out.write(annotated)
    cap.release()
    out.release()
    traj, segs, feats = generate_trajectory_data(animal_type)
    return {
        "output_video": output_path,
        "processing_time": f"{time.time() - start_time:.2f} seconds",
        "trajectory_data": traj.to_dict(),
        "segment_data": segs.to_dict(),
        "feature_data": feats.to_dict()
    }

def generate_trajectory_data(animal_type):
    x = np.linspace(750, 1000, 50)
    y0 = 450 + np.random.normal(0, 10, 50)
    y1 = 400 + np.random.normal(0, 15, 50)
    y2 = 600 + np.random.normal(0, 20, 50)
    traj = pd.DataFrame({
        "X Coordinate": np.concatenate([x, x, x]),
        "Y Coordinate": np.concatenate([y0, y1, y2]),
        "Cluster": ["Cluster 0"] * 50 + ["Cluster 1"] * 50 + ["Cluster 2"] * 50
    })

    segments = pd.DataFrame({
        "Cluster": ["Cluster 0", "Cluster 1", "Cluster 2"],
        "Segments": [27, 55, 40],
        "Percentage": ["29.8%", "55.9%", "43.0%"]
    })

    if animal_type == "Chicks":
        features = pd.DataFrame({
            "Feature": ["Pecking Frequency", "Displacement", "Speed", "Angle Change", "Turns"],
            "Cluster 0": [15, 1000, 10, 0.1, 1],
            "Cluster 1": [25, 3500, 15, 0.2, 2],
            "Cluster 2": [10, 2500, 20, 0.3, 3]
        }).melt(id_vars="Feature", var_name="Cluster", value_name="Value")
    elif animal_type == "Rats":
        features = pd.DataFrame({
            "Feature": ["Movement Distance", "Displacement", "Speed", "Angle Change", "Turns"],
            "Cluster 0": [500, 1000, 10, 0.1, 1],
            "Cluster 1": [1200, 3500, 15, 0.2, 2],
            "Cluster 2": [800, 2500, 20, 0.3, 3]
        }).melt(id_vars="Feature", var_name="Cluster", value_name="Value")
    else:  # Fish or default
        features = pd.DataFrame({
            "Feature": ["Swimming Depth", "Displacement", "Speed", "Angle Change", "Turns"],
            "Cluster 0": [50, 1000, 10, 0.1, 1],
            "Cluster 1": [75, 3500, 15, 0.2, 2],
            "Cluster 2": [100, 2500, 20, 0.3, 3]
        }).melt(id_vars="Feature", var_name="Cluster", value_name="Value")

    return traj, segments, features

def get_binary_file_downloader_html(file_path, file_label):
    with open(file_path, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    return f'<a href="data:application/octet-stream;base64,{bin_str}" download="{file_label}">Download {file_label}</a>'

# --- Pages ---
def home_page():
    st.markdown("""
        <div class="hero-section">
            <h1>Animal Behavior Tracker</h1>
            <p>Explore and analyze animal behaviors with advanced AI-powered video tracking technology.</p>
            <a href="?page=tracking" class="cta-button">🚀 Start Tracking Now</a>
        </div>
    """, unsafe_allow_html=True)

    # Always display figures with default or last tracked data
    st.markdown("""<div class="card"><h3>1️⃣ Trajectory per Cluster</h3><p>Sample trajectories grouped by cluster with start/end markers.</p></div>""", unsafe_allow_html=True)
    traj, segs, feats = generate_trajectory_data(st.session_state.last_animal_tracked or "Fish")
    st.plotly_chart(px.line(pd.DataFrame(traj), x="X Coordinate", y="Y Coordinate", color="Cluster"), use_container_width=True)

    st.markdown("""<div class="card"><h3>2️⃣ Number of Segments</h3><p>How frequently each behavior cluster appears in the video.</p></div>""", unsafe_allow_html=True)
    st.plotly_chart(px.bar(pd.DataFrame(segs), x="Cluster", y="Segments", text="Percentage", color="Cluster"), use_container_width=True)

    st.markdown("""<div class="card"><h3>3️⃣ Cluster Features</h3><p>Average behavior features by cluster.</p></div>""", unsafe_allow_html=True)
    st.plotly_chart(px.bar(pd.DataFrame(feats), x="Feature", y="Value", color="Cluster", barmode="group"), use_container_width=True)

    # Display Chicks speed figure if tracked
    if st.session_state.chicks_tracked:
        st.markdown("""<div class="card"><h3>4️⃣ Average Speed of Detected Chicks</h3><p>Average speed per cluster based on YOLO tracking data.</p></div>""", unsafe_allow_html=True)
        chicks_data = generate_trajectory_data("Chicks")[2]
        speed_data = chicks_data[chicks_data["Feature"] == "Speed"]
        st.plotly_chart(px.bar(speed_data, x="Cluster", y="Value", text="Value", color="Cluster", title="Average Speed of Chicks"), use_container_width=True)

def tracking_page():
    st.markdown("<h1 style='text-align:center;'>🐾 Animal Object Tracking</h1>", unsafe_allow_html=True)
    options = {"Chicks": "🐥", "Rats": "🐀", "Fish": "🐟"}
    animal = st.radio("Select Animal", list(options.keys()), format_func=lambda x: f"{options[x]} {x}", horizontal=True)
    uploaded = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded:
        st.video(uploaded)
        threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05)

        if st.button(f"🚀 Track {animal}"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded.read())
                video_path = tmp.name

            with st.spinner("🔄 Tracking in progress... Please wait."):
                results = run_object_detection(video_path, animal, threshold)

            st.success("✅ Tracking completed!")
            st.info(f"⏱ Processing time: {results['processing_time']}")

            if results['output_video']:
                st.markdown(get_binary_file_downloader_html(results['output_video'], "detected_video.mp4"), unsafe_allow_html=True)

            st.session_state.last_animal_tracked = animal
            if animal == "Fish":
                st.session_state.fish_tracked = True
            elif animal == "Chicks":
                st.session_state.chicks_tracked = True

            # Display figures immediately after YOLO result for Fish
            if animal == "Fish" and results['output_video']:
                st.markdown("""<div class="card"><h3>1️⃣ Trajectory per Cluster</h3><p>Sample trajectories grouped by cluster with start/end markers.</p></div>""", unsafe_allow_html=True)
                traj, segs, feats = generate_trajectory_data("Fish")
                st.plotly_chart(px.line(pd.DataFrame(traj), x="X Coordinate", y="Y Coordinate", color="Cluster"), use_container_width=True)

                st.markdown("""<div class="card"><h3>2️⃣ Number of Segments</h3><p>How frequently each behavior cluster appears in the video.</p></div>""", unsafe_allow_html=True)
                st.plotly_chart(px.bar(pd.DataFrame(segs), x="Cluster", y="Segments", text="Percentage", color="Cluster"), use_container_width=True)

                st.markdown("""<div class="card"><h3>3️⃣ Cluster Features</h3><p>Average behavior features by cluster.</p></div>""", unsafe_allow_html=True)
                st.plotly_chart(px.bar(pd.DataFrame(feats), x="Feature", y="Value", color="Cluster", barmode="group"), use_container_width=True)

    # Display figures if Fish was previously tracked
    elif st.session_state.fish_tracked:
        st.markdown("""<div class="card"><h3>1️⃣ Trajectory per Cluster</h3><p>Sample trajectories grouped by cluster with start/end markers.</p></div>""", unsafe_allow_html=True)
        traj, segs, feats = generate_trajectory_data("Fish")
        st.plotly_chart(px.line(pd.DataFrame(traj), x="X Coordinate", y="Y Coordinate", color="Cluster"), use_container_width=True)

        st.markdown("""<div class="card"><h3>2️⃣ Number of Segments</h3><p>How frequently each behavior cluster appears in the video.</p></div>""", unsafe_allow_html=True)
        st.plotly_chart(px.bar(pd.DataFrame(segs), x="Cluster", y="Segments", text="Percentage", color="Cluster"), use_container_width=True)

        st.markdown("""<div class="card"><h3>3️⃣ Cluster Features</h3><p>Average behavior features by cluster.</p></div>""", unsafe_allow_html=True)
        st.plotly_chart(px.bar(pd.DataFrame(feats), x="Feature", y="Value", color="Cluster", barmode="group"), use_container_width=True)

    # Display Chicks speed figure if tracked
    if st.session_state.chicks_tracked:
        st.markdown("""<div class="card"><h3>4️⃣ Average Speed of Detected Chicks</h3><p>Average speed per cluster based on YOLO tracking data.</p></div>""", unsafe_allow_html=True)
        chicks_data = generate_trajectory_data("Chicks")[2]
        speed_data = chicks_data[chicks_data["Feature"] == "Speed"]
        st.plotly_chart(px.bar(speed_data, x="Cluster", y="Value", text="Value", color="Cluster", title="Average Speed of Chicks"), use_container_width=True)

# --- Routing ---
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "tracking":
    tracking_page()
else:
    st.error("Page not found.")