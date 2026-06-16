import cv2
import numpy as np
from src.model import TrafficModel

class DetectionPipeline:
    def __init__(self, model: TrafficModel, config: dict):
        self.model = model
        self.config = config
        self.class_names = ['Red Light', 'Yellow Light', 'Green Light']

    def run(self):
        video_src = self.config['video']['source_path']
        cap = cv2.VideoCapture(video_src)

        if not cap.isOpened():
            print(f"❌ Error: Cannot open video source {video_src}")
            return
        
        cv2.namedWindow("Smart Traffic Signal Assistant", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Smart Traffic Signal Assistant", 800, 600)

        print("🚀 Starting real-time detection pipeline. Press 'q' to exit.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
           
            results = self.model.predict(
                frame,
                imgsz=self.config['model']['img_size'],
                conf=self.config['model']['confidence_threshold']
            )
            
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    conf_score = float(box.conf[0])
                    
                    # 🧠 Data Science Guardrail: Correct the model if it hallucinates "Red"
                    if cls_id == 0:  
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                
                            # Broad range to capture green light emission shades
                            lower_green = np.array([35, 40, 40])
                            upper_green = np.array([85, 255, 255])
                            green_mask = cv2.inRange(hsv, lower_green, upper_green)
                
                            # If more than 6% of the bounding box is green, force change the class ID to 2
                            if (cv2.countNonZero(green_mask) / crop.size) > 0.06:
                                cls_id = 2  # Override to Green Light

                    label = self.class_names[cls_id]

                    if label == 'Red Light':
                        color = (0, 0, 255)
                        alert_text = "ALERT: RED LIGHT - STOP!"
                    elif label == 'Yellow Light':
                        color = (0, 255, 255)
                        alert_text = "WARNING: CLEAR THE INTERSECTION!"
                    else:
                        color = (0, 255, 0)
                        alert_text = "STATUS: GREEN LIGHT - GO"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {conf_score:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    cv2.putText(frame, alert_text, (30, 50), 
                                cv2.FONT_HERSHEY_DUPLEX, 
                                self.config['alerts']['font_scale'], color, 
                                self.config['alerts']['thickness'])

            if self.config['video']['show_window']:
                cv2.imshow("Smart Traffic Signal Assistant", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
        print("🏁 Video pipeline finished successfully.")