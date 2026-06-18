import streamlit as st
import os
import yaml
import shutil
from src.pipeline import TrafficLightPipeline

# Page Setup
st.set_page_config(layout="wide")
st.title("🚦 Balanced Traffic Light Detection System")

# Load Config
config_path = "config.yaml"
with open(config_path, 'r') as file:
    current_config = yaml.safe_load(file)

# Columns Layout
col1, col2 = st.columns([1, 2])

# Col 1: Controls
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
            pipeline = TrafficLightPipeline(config_path=config_path)
            pipeline.process_video()
            
            # Persist data
            st.session_state.last_output_path = pipeline.output_path 
            st.session_state.pipeline = pipeline
            st.success("✅ Processing complete!")

# Col 2: Display
with col2:
    st.subheader("📊 Analytical Output Display")
    
    # Check if pipeline data exists in session
    pipeline = st.session_state.get('pipeline')
    output_path = st.session_state.get('last_output_path')
    
    if pipeline and output_path and os.path.exists(output_path):
        # 1. Display Timeline Alerts
        st.info("### 🕒 Detection Timeline")
        for event in pipeline.status_history:
            status_icon = "🚨" if "STOP" in event['status'] else "✅"
            st.write(f"⏱️ **{event['time']}s**: {status_icon} {event['status']}")
        
        # 2. Display Processed Video
        streaming_path = "Data/test_videos/stream_ready.mp4"
        shutil.copyfile(output_path, streaming_path)
        st.video(streaming_path, format="video/mp4")
    else:
        st.info("Upload a video and click 'Execute' to begin analysis.")