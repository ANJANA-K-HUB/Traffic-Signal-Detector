import os
from ultralytics import YOLO

def main():
    # Load a fresh, lightweight pretrained YOLOv8 nano model
    model = YOLO("yolov8n.pt") 

    #  Automatically find the absolute path to your local data.yaml config
    yaml_path = os.path.join(os.getcwd(), "TrafficLightData", "data.yaml")

    print(f"\nStarting training using data configuration: {yaml_path}\n")

    # Train the model 
    model.train(
        data=yaml_path,
        epochs=40,         
        imgsz=640,         
        batch=16,          
        device=0,          
        workers=2,         
        augment=True       
    )

if __name__ == "__main__":
    main()