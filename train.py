from ultralytics import YOLO

def train_model():
    print("🚀 Initializing YOLOv8 Nano Model...")
    # Load a pre-trained YOLOv8 nano model (lightweight & fast)
    model = YOLO("yolov8n.pt") 
    
    print("🏋️ Starting training process...")
    # Train the model using our custom dataset config
    model.train(
        data="data.yaml", 
        epochs=50,       # Number of training rounds (adjust based on time/accuracy needs)
        imgsz=640,       # Standard YOLO image resolution
        device="cpu"     # Uses CPU. Change to 0 or 'cuda' if you have an NVIDIA GPU setup
    )
    print("✅ Training complete! Weights saved to: runs/detect/train/weights/best.pt")

if __name__ == "__main__":
    train_model()