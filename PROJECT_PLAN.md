# 🛡️ SENTRY: Autonomous Real-Time Weapon Detection System

![Project Status](https://img.shields.io/badge/Status-Active_Development-green)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Model](https://img.shields.io/badge/Model-YOLOv8-orange)
![Deployment](https://img.shields.io/badge/Frontend-Streamlit-red)

## 📖 Project Overview
**Sentry** is an active computer vision security system designed to detect hostile threats (handguns, rifles, knives) in real-time video feeds. Unlike passive CCTV systems that only record footage, Sentry analyzes frames using deep learning to provide instant alerts to security personnel, potentially reducing response times during critical incidents.

This project is built for a Capstone Portfolio, demonstrating proficiency in **Edge AI**, **Object Detection**, and **Full-Stack AI Deployment**.

---

## 🎯 Key Features
- **Real-Time Detection:** Processes video streams at >30 FPS using YOLOv8-Nano.
- **Multi-Class Recognition:** Identifies `Pistol`, `Rifle`, and `Knife`.
- **Confidence Filtering:** Adjustable sensitivity threshold to minimize false positives (e.g., confusing a phone with a gun).
- **Incident Logging:** Automatically captures timestamps and screenshots of detected threats.
- **Visual Alert System:** Dynamic dashboard UI that flashes "RED" when a threat is confirmed.

---

## 🛠️ Tech Stack & Architecture

### **Core AI Engine**
* **Model:** [YOLOv8 (You Only Look Once)](https://github.com/ultralytics/ultralytics)
* **Framework:** PyTorch
* **Training Platform:** Google Colab (T4 GPU)

### **Dataset**
* **Source:** Roboflow Universe (`weapon-detection-pgqnr`)
* **Size:** ~25,000 Images
* **Classes:** `pistol`, `rifle`, `knife` (and potentially `smartphone` as negative class)
* **Preprocessing:** Auto-orientation, resize to 640x640.

### **Deployment Application**
* **Frontend:** [Streamlit](https://streamlit.io/)
* **Image Processing:** OpenCV (`cv2`)
* **Notification:** (Optional) Twilio API / Discord Webhook for SMS alerts.

---

## 🗺️ Project Roadmap

### **Phase 1: Data Preparation & Training** (Week 1)
- [x] **Data Sourcing:** Download `weapon-detection-pgqnr` from Roboflow.
- [ ] **Data Audit:** Verify class balance (ensure enough examples of pistols vs. rifles).
- [ ] **Model Training:**
    - Train `yolov8n.pt` (Nano) for speed.
    - Set epochs: 25-50 (Early Stopping enabled).
    - Image Size: 640.
- [ ] **Evaluation:** Analyze Confusion Matrix and F1-Score to check for False Positives.

### **Phase 2: Core Logic Development** (Week 2) ✅
- [x] Develop `detect.py` script.
- [x] Implement video stream ingestion (Webcam/File).
- [x] Create logic for "Threat Persistence" (e.g., only alert if weapon seen for 5 consecutive frames to avoid glitches).

### **Phase 3: User Interface (Streamlit)** (Week 3) ✅
- [x] Build Dashboard Layout (Video Feed Left, Logs Right).
- [x] Add Sidebar Controls (Confidence Slider, Model Selector).
- [x] Implement "Red Alert" visual state.
- [x] Build "Incident Log" dataframe that updates live.

### **Phase 4: Testing & Documentation** (Week 4)
- [ ] **Stress Test:** Test with low-light footage and blurry motion.
- [ ] **False Positive Test:** Test with black smartphones and wallets.
- [ ] **Final Polish:** Clean up code, add comments, and finalize this README.

---

## 📂 Repository Structure

```bash
Sentry/
├── data/
│   ├── sample_video.mp4       # For testing without webcam
│   └── screenshots/           # Saved images of detected threats
├── models/
│   └── sentry_best.pt         # The trained YOLOv8 weights
├── src/
│   ├── app.py                 # Main Streamlit Dashboard script
│   └── utils.py               # Helper functions (logging, alerts)
├── notebooks/
│   └── Sentry_Training.ipynb  # Google Colab notebook used for training
├── requirements.txt           # Python dependencies
└── README.md                  # Project Documentation