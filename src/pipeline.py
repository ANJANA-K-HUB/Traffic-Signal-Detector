import cv2
import os
import yaml
import ffmpeg
import shutil
import datetime
from ultralytics import YOLO

print("--- DEBUG: pipeline.py is being loaded ---")
class TrafficLightPipeline:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        self.model = YOLO(self.config['model']['weights_path'])
        self.source_path = self.config['video']['source_path']
        self.output_path = self.config['video']['output_path']
        self.conf_threshold = self.config['model']['confidence_threshold']
        
        # This is the attribute that was missing!
        self.status_history = [] 
      

    def process_video(self):
        # Generate a unique output path
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"output_{timestamp}.mp4"
        self.output_path = os.path.join(os.path.dirname(self.output_path), unique_filename)
        
        cap = cv2.VideoCapture(self.source_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        
        temp_dir = "temp_frames"
        os.makedirs(temp_dir, exist_ok=True)
        
        last_recorded_status = None
        count = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success: 
                break
           
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            
            if results[0].boxes:
                class_id = int(results[0].boxes[0].cls)
                
                if class_id == 0:
                    current_status = "STOP (Red)"
                elif class_id == 1:
                    current_status = "GO (Green)"
                elif class_id == 2:
                    current_status = "CAUTION (Yellow)"
                else:
                    current_status = "Unknown"
            # ----------------------------------------
                

                
                if current_status != last_recorded_status:
                    time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    self.status_history.append({"time": round(time_sec, 2), "status": current_status})
                    last_recorded_status = current_status
            
            cv2.imwrite(f"{temp_dir}/frame_{count:05d}.jpg", results[0].plot())
            count += 1
            
        cap.release()

        try:
            (ffmpeg.input(f'{temp_dir}/frame_%05d.jpg', framerate=fps)
             .output(self.output_path, vcodec='libx264', pix_fmt='yuv420p', crf=23)
             .overwrite_output().run(quiet=True))
        finally:
            shutil.rmtree(temp_dir)