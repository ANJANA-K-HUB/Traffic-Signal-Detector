import sys
from src.pipeline_new import TrafficLightPipeline

def main():
    print("Initializing Balanced Traffic Light Detection System...")
    
    try:
        # Initialize the processing pipeline with your configuration settings
        pipeline = TrafficLightPipeline(config_path="config.yaml")
        
        print("Starting video frame-by-frame inference pipeline...")
        pipeline.process_video()
        
        print("Video processing completed successfully! Check your configured output path.")
        
    except Exception as e:
        print(f"An error occurred during pipeline execution: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()