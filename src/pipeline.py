import os
import yaml
import cv2
from ultralytics import YOLO

class TrafficLightPipeline:
    def __init__(self, config_path="config.yaml"):
        """Initializes the pipeline by loading configurations and model weights."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        # Load Model Configurations
        self.weights_path = self.config['model']['weights_path']
        self.conf_threshold = self.config['model']['confidence_threshold']
        self.img_size = self.config['model']['img_size']
        
        # Load Video Configurations
        self.source_path = self.config['video']['source_path']
        self.output_path = self.config['video']['output_path']
        self.show_window = self.config['video']['show_window']
        
        # Initialize YOLO Model
        self.model = YOLO(self.weights_path)
        
        # TRUST THE MODEL'S INTERNAL DICTIONARY NAMES DIRECTLY
        self.class_names = self.model.names  
        print(f"Model classes loaded dynamically: {self.class_names}")

    def process_video(self):
        """Runs object detection frame-by-frame and displays high-visibility status alerts."""
        if not os.path.exists(self.source_path):
            raise FileNotFoundError(f"Input video not found at: {self.source_path}")
            
        cap = cv2.VideoCapture(self.source_path)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        win_title = "Traffic Light Pipeline"
        if self.show_window:
            cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            results = self.model.predict(
                source=frame, 
                conf=self.conf_threshold, 
                imgsz=self.img_size, 
                rect=True,         
                iou=0.45,          
                verbose=False
            )
            
            # Variables to track what colors are present in the current frame
            detected_colors = set()
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    raw_label = self.class_names.get(cls_id, "").lower()

                    if 'red' in raw_label:
                        color = (0, 0, 255)       # BGR: Red
                        display_text = f"Red Light: {conf:.2f}"
                        detected_colors.add("red")
                    elif 'yellow' in raw_label:
                        color = (0, 255, 255)     # BGR: Yellow
                        display_text = f"Yellow Light: {conf:.2f}"
                        detected_colors.add("yellow")
                    elif 'green' in raw_label:
                        color = (0, 255, 0)       # BGR: Green
                        display_text = f"Green Light: {conf:.2f}"
                        detected_colors.add("green")
                    else:
                        color = (255, 255, 255)   
                        display_text = f"Class {cls_id}: {conf:.2f}"
                        
                    # Draw object bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, display_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # --- HIGH-VISIBILITY ACTION ALERT LOGIC ---
            if "red" in detected_colors:
                alert_text = "ALERT: STOP"
                alert_color = (0, 0, 255)       # Bright Red text
            elif "yellow" in detected_colors:
                alert_text = "ALERT: READY"
                alert_color = (0, 255, 255)     # Bright Yellow text
            elif "green" in detected_colors:
                alert_text = "ALERT: GO"
                alert_color = (0, 255, 0)       # Bright Green text
            else:
                alert_text = "STATUS: CLEAR"
                alert_color = (255, 255, 255)   # White text
                
            # Expanded black background container to neatly hold the larger text
            cv2.rectangle(frame, (20, 20), (460, 95), (0, 0, 0), -1)
            
            # Rendered with fontScale=1.4 and thickness=3 for maximum readability
            cv2.putText(frame, alert_text, (35, 70),
                        cv2.FONT_HERSHEY_DUPLEX, 1.4, alert_color, 3, cv2.LINE_AA)
            
            # Save high-resolution quality to output file
            out.write(frame)
            
            if self.show_window:
                # Keep preview window cleanly sized to fit your monitor screen
                preview_width = 854
                preview_height = 480
                resized_preview = cv2.resize(frame, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
                
                cv2.imshow(win_title, resized_preview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    pipeline = TrafficLightPipeline(config_path="config.yaml")
    pipeline.process_video()