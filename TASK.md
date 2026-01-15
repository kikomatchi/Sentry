# Phase 1: Data Preparation & Training (Plan)

This document outlines the detailed steps to complete Phase 1 of the Sentry Project.

## 1. Environment Setup
- [ ] **Create Virtual Environment** (Optional but recommended)
- [ ] **Install Dependencies**
    - Create `requirements.txt` with:
        - `ultralytics` (YOLOv8)
        - `torch`, `torchvision` (if not pulled by ultralytics)
        - `opencv-python`
        - `pandas`
        - `matplotlib`
        - `notebook` or `jupyterlab`
    - Run `pip install -r requirements.txt`

## 2. Dataset Preparation
- [ ] **Unpack Dataset**
    - Extract `Weapon-Detection.v8-adi_v2.yolov8.zip` into the `data/` directory.
    - Expected structure:
        ```text
        data/
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── valid/ (or test/)
        │   ├── images/
        │   └── labels/
        └── data.yaml
        ```
- [ ] **Configure `data.yaml`**
    - Verify that the paths in `data.yaml` point correctly to the local `train` and `valid` directories. Use absolute paths if necessary.

## 3. Data Audit
- [ ] **Create Audit Notebook** (`notebooks/01_Data_Audit.ipynb`)
    - **Class Distribution Analysis**: Write a script to parse all label files and count instances of `pistol`, `rifle`, `knife`.
    - **Visual Inspection**: Display a random batch of 10-20 images with their bounding boxes drawn to ensure labels match targets.
    - **Resolution Check**: Confirm image sizes (mostly 640x640 expected).

## 4. Model Training
- [ ] **Create Training Notebook** (`notebooks/Sentry_Training.ipynb`)
    - **Load Model**: Initialize `YOLO('yolov8n.pt')`.
    - **Training Configuration**:
        - `data`: path to `data.yaml`
        - `epochs`: 50 (with `patience=10` for early stopping)
        - `imgsz`: 640
        - `batch`: 16 (adjust based on GPU memory, e.g., 8 or 32)
        - `project`: `../models`
        - `name`: `sentry_phase1`
    - **Execution**: Run the training loop.

## 5. Evaluation & Validation
- [ ] **Analyze Training Metrics**
    - Review `results.csv` generated in the run folder.
    - Check loss curves (box_loss, cls_loss).
    - **Key Metrics**: mAP@50 and mAP@50-95.
- [ ] **Confusion Matrix Analysis**
    - Inspect the confusion matrix to see if classes are being confused (e.g., Rifle vs Pistol).
- [ ] **Inference Test**
    - Run inference on the `valid` set images and visualize results.
    - (Optional) detailed analysis of False Positives.
- [ ] **Save Constraints**
    - Save the best model as `models/sentry_best.pt`.

## 6. Documentation
- [ ] Update `PROJECT_PLAN.md` with actual results (mAP scores).
- [ ] Document any dataset issues found during the audit.
