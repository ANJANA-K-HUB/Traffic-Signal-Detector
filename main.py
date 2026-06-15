import yaml
from src.model import TrafficModel
from src.pipeline import DetectionPipeline

def main():
    # Load configuration parameters
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize components
    model = TrafficModel(weights_path=config['model']['weights_path'])
    pipeline = DetectionPipeline(model=model, config=config)

    # Run pipeline
    pipeline.run()

if __name__ == "__main__":
    main()