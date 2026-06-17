import cv2
import os
import yaml
import ffmpeg
import shutil
from ultralytics import YOLO

class TrafficLightPipeline:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        self.model = YOLO(self.config['model']['weights_path'])
        self.source_path = self.config['video']['source_path']
        self.output_path = self.config['video']['output_path']
        self.conf_threshold = self.config['model']['confidence_threshold']
        
        # THIS IS THE MISSING PART:
        self.final_status = "Scanning..." 

    def process_video(self):
        cap = cv2.VideoCapture(self.source_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        
        temp_dir = "temp_frames"
        os.makedirs(temp_dir, exist_ok=True)
        
        print("Processing frames...")
        count = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            
            # Logic: Update the attribute
            if results[0].boxes:
                class_id = int(results[0].boxes[0].cls)
                self.final_status = "STOP (Red)" if class_id == 0 else "GO (Green)"
            
            cv2.imwrite(f"{temp_dir}/frame_{count:05d}.jpg", results[0].plot())
            count += 1
            
        cap.release()

        # Finalize video
        try:
            (ffmpeg.input(f'{temp_dir}/frame_%05d.jpg', framerate=fps)
             .output(self.output_path, vcodec='libx264', pix_fmt='yuv420p', crf=23)
             .overwrite_output().run(quiet=True))
        finally:
            shutil.rmtree(temp_dir)