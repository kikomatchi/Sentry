# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sentry is a real-time weapon detection system using YOLOv8 for identifying handguns, rifles, and knives in video feeds. It's a capstone project demonstrating Edge AI, object detection, and full-stack AI deployment.

## Tech Stack

- **Language:** Python 3.10
- **AI Framework:** PyTorch + YOLOv8 (Nano variant for speed)
- **Frontend:** Streamlit
- **Computer Vision:** OpenCV (cv2)
- **Training Platform:** Google Colab (T4 GPU)
- **Dataset:** Roboflow Universe (`weapon-detection-pgqnr`) - ~25k images, 640x640

## Commands

```bash
# Create and activate virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit dashboard
streamlit run src/app.py
```

## Architecture

```
Sentry/
├── src/
│   ├── app.py          # Main Streamlit dashboard
│   └── utils.py        # Helper functions (logging, alerts)
├── models/
│   └── sentry_best.pt  # Trained YOLOv8 weights
├── data/
│   ├── sample_video.mp4
│   └── screenshots/    # Captured threat images
├── notebooks/
│   └── Sentry_Training.ipynb  # Colab training notebook
└── requirements.txt
```

## Key Implementation Details

- **Detection Classes:** `pistol`, `rifle`, `knife` (optional `smartphone` as negative class)
- **Performance Target:** >30 FPS using YOLOv8-Nano
- **Threat Persistence:** Alert only after weapon detected in 5 consecutive frames (prevents false triggers)
- **Model Training:** 25-50 epochs with early stopping, image size 640

## Core Dependencies

- ultralytics (YOLOv8)
- torch, torchvision
- opencv-python
- streamlit
- numpy, pandas
- (Optional) twilio or discord.py for notifications
