# 🚦 AI-Powered Traffic Signal Monitoring & Automation System

An automated computer vision pipeline designed for **real-time traffic signal state tracking and cycle auditing**. The system processes raw traffic video footage, detects traffic lights using a trained YOLOv8 model, identifies their operational state (**Red, Green, Yellow**), and exports an event-driven timeline of signal transitions for analysis and auditing.

---

## 📌 Features

### 🔴🟢🟡 Ternary Signal State Tracking
Detects and classifies traffic light states into:

- **STOP (Red)**
- **GO (Green)**
- **CAUTION (Yellow)**

### ⏱️ Event-Driven Timeline Logging
Records timestamps only when a genuine signal transition occurs, creating a concise and accurate audit log.

### 🛡️ Robust State Latching
Implements fallback and persistence logic to reduce false transitions caused by temporary occlusions, glare, or missed detections.

### 📊 Interactive Streamlit Dashboard
Provides a user-friendly web interface for:

- Uploading traffic videos
- Running signal analysis
- Viewing detected signal states
- Downloading generated timeline logs

### ⚡ High-Speed Inference
Optimized YOLOv8-based detection pipeline capable of near real-time processing performance.

---

# 📂 Project Structure

```text
TRAFFICSIGNAL/
│
├── app.py                  # Streamlit Dashboard Interface
├── main.py                 # CLI Entry Point
├── data.yaml               # Dataset Configuration
├── requirements.txt        # Project Dependencies
├── .gitignore              # Git Ignore Rules
│
├── src/
│   ├── __init__.py
│   └── pipeline.py         # Detection & Tracking Pipeline
│
└── weights/
    └── best.pt             # Trained YOLOv8 Model Weights
```

---

# 📊 Dataset Information

### Dataset Source

- Roboflow Prescan v6
- Traffic signal image dataset

### Dataset Statistics

| Metric | Value |
|----------|----------|
| Total Images | 4,969 |
| Image Resolution | 416 × 416 |
| Classes | Red, Green, Yellow |

---

# 📈 Model Performance

## Validation Metrics

| Class | Images | Instances | Precision (P) | Recall (R) |
|---------|---------|---------|---------|---------|
| All Classes | 50 | 620 | 0.960 | 0.971 |
| Green Light | 24 | 321 | 1.000 | 0.912 |
| Red Light | 20 | 240 | 0.996 | 1.000 |
| Yellow Light | 6 | 60 | 0.883 | 1.000 |

---

## ⚡ Inference Performance

| Metric | Value |
|----------|----------|
| Neural Inference Time | 9.1 ms |
| End-to-End Processing | 10.1 ms |
| Estimated Throughput | ~99 FPS |

---

# 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/TrafficSignal.git
cd TrafficSignal
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

### Windows

```bash
.\venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🚀 Usage

## Option 1: Launch Streamlit Dashboard

```bash
streamlit run app.py
```

Open the displayed local URL in your browser and upload a traffic video for analysis.

---

## Option 2: Run via Command Line

```bash
python main.py --source data/raw/sample_traffic.mp4
```

Replace the source path with your desired video file.

---

# ⚙️ Core Detection Logic

The system continuously performs frame-by-frame inference and maps YOLO class predictions to traffic signal states.

```python
if results[0].boxes:
    class_id = int(results[0].boxes[0].cls)

    if class_id == 0:
        current_status = "STOP (Red)"
    elif class_id == 1:
        current_status = "GO (Green)"
    elif class_id == 2:
        current_status = "CAUTION (Yellow)"
else:
    current_status = "None"
```

State transitions are logged only when the detected state changes, creating an efficient event-driven timeline.

---

# 📄 Example Timeline Output

```text
00:00:02  STOP (Red)
00:00:15  GO (Green)
00:00:28  CAUTION (Yellow)
00:00:31  STOP (Red)
```

---

# 🔧 Technology Stack

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Pandas
- NumPy
- Streamlit

---

# 🎯 Applications

- Smart Traffic Monitoring
- Traffic Signal Auditing
- Urban Infrastructure Analytics
- Intelligent Transportation Systems (ITS)
- Traffic Flow Research
- Smart City Deployments

---
