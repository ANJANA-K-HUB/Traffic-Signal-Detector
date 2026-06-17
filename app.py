import streamlit as st
import os
import yaml
import shutil
from src.pipeline import TrafficLightPipeline

# 1. Page Configuration
st.set_page_config(page_title="Traffic Light System", layout="wide")
st.title("🚦 Balanced Traffic Light Detection System")

# 2. Config Loading
config_path = "config.yaml"
with open(config_path, 'r') as file:
    current_config = yaml.safe_load(file)

# We use Session State so the 'status' persists after the script reruns
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None

# 3. Layout Columns
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("⚙️ System Controls")
    uploaded_file = st.file_uploader("Choose a traffic video...", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        source_path = current_config['video']['source_path']
        with open(source_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File uploaded successfully!")

    if st.button("🚀 Execute Detection Pipeline"):
        with st.spinner("Processing video..."):
            try:
                # Initialize and run
                pipeline = TrafficLightPipeline(config_path=config_path)
                pipeline.process_video()
                # Save to session state
                st.session_state.pipeline = pipeline 
                st.success("✅ Processing complete!")
            except Exception as e:
                st.error(f"Error: {e}")

with col2:
    st.subheader("📊 Analytical Output Display")
    output_path = current_config['video']['output_path']
    
    if os.path.exists(output_path):
        # Retrieve pipeline from session state
        pipeline = st.session_state.pipeline
        if pipeline:
            status = pipeline.final_status
            if "STOP" in status:
                st.error(f"🚨 ALERT: {status}")
            else:
                st.success(f"✅ STATUS: {status}")
        
        # Stream the video
        streaming_path = "Data/test_videos/stream_ready.mp4"
        shutil.copyfile(output_path, streaming_path)
        st.video(streaming_path, format="video/mp4")
    else:
        st.info("Upload a video and click 'Execute' to begin.")