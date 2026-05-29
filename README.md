# Golf Ball Tracker

Computer vision project for detecting and tracking a golf ball in golf swing videos.

The project uses a YOLO-based object detector together with ROI-based tracking, candidate filtering and Kalman-based motion prediction to follow the golf ball after impact. The final result is an annotated video or image sequence showing the detected ball position and the estimated ball trajectory.

---

## Project Goal

The goal of this project is to detect and track a golf ball in video footage after a golf swing.

This is challenging because:

- the golf ball is very small in the frame
- the ball moves very fast after impact
- motion blur makes detection difficult
- the ball can disappear for several frames
- false detections can appear in the background
- full-frame detection alone is not reliable enough after launch

To solve this, the project combines object detection with tracking logic.

---

## Important Note: Dataset and Model Weights

This repository does **not** include the full dataset, extracted frames, input videos or trained YOLO model weights.

These files are not included because they can be large and are not suitable for direct upload to GitHub.

To run the project, the user needs to provide:

```text
1. Input video files or extracted frames
2. A trained YOLO model file, for example best.pt
3. Correct local paths inside the detection script
````

For the original project demo, the dataset, extracted frames and trained model weights were stored locally and used for presentation purposes.

---

## Repository Structure

```text
golf-ball-tracker/
│
├── src/
│   ├── detect_video.py              # Main detection and tracking pipeline
│   ├── tracker.py                   # Kalman-based ball tracker
│   ├── overlay.py                   # Creates trajectory overlay / final visualisation
│   ├── train_detector.py            # YOLO training script
│   ├── extract_frames.py            # Extracts frames from input videos
│   ├── convert_rucv_voc_to_yolo.py  # Converts RUCV VOC annotations to YOLO format
│   ├── split_rucv_dataset.py        # Splits dataset into train/valid/test
│   ├── evaluate.py                  # Evaluation script
│   ├── pipeline.py                  # Pipeline file
│   └── roi_detection.py             # ROI detection helper file
│
├── README.md
├── requirements.txt
└── .gitignore
```

Large local folders such as `data/`, `models/`, `runs/` and `output/` are not included in the repository.

---

## Expected Local File Structure

To run the project locally, the required data and model files should be placed in a structure similar to this:

```text
golf-ball-tracker/
│
├── data/
│   ├── videos/
│   │   └── example_video.mp4
│   │
│   └── extracted_frames/
│       └── example_video/
│           ├── frame_00000.jpg
│           ├── frame_00001.jpg
│           ├── frame_00002.jpg
│           └── ...
│
├── models/
│   └── best.pt
│
├── output/
│
└── src/
```

The exact paths can be adjusted inside the Python scripts, especially in:

```text
src/detect_video.py
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/KimBor04/golf-ball-tracker.git
cd golf-ball-tracker
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Requirements

The main dependencies are:

```text
ultralytics
opencv-python
numpy
matplotlib
torch
torchvision
pillow
PyYAML
requests
scipy
```

These should be listed in `requirements.txt`.

---

## Model Weights

The project uses a YOLO model trained for golf ball detection.

The trained model weights are not included in this repository. Model files such as `.pt` files are ignored because they can be large.

Example model file:

```text
models/best.pt
```

or:

```text
runs/detect/models/experiments/yolo11n_post_impact_12803/weights/best.pt
```

The model path must match the path used inside `src/detect_video.py`.

Example:

```python
model_path = "models/best.pt"
```

or:

```python
model_path = "runs/detect/models/experiments/yolo11n_post_impact_12803/weights/best.pt"
```

If the model file is missing, the detection script cannot run.

---

## Input Data

The tracker expects either a video file that can be converted into frames or already extracted image frames.

Example extracted frame folder:

```text
data/extracted_frames/rory_005/
```

Example frame files:

```text
data/extracted_frames/rory_005/frame_00000.jpg
data/extracted_frames/rory_005/frame_00001.jpg
data/extracted_frames/rory_005/frame_00002.jpg
```

The input frame path must match the path used inside `src/detect_video.py`.

---

## Extract Frames from a Video

Use the frame extraction script to convert a video into individual frames:

```bash
python src/extract_frames.py --video data/videos/example_video.mp4 --every-n 1
```

The `--every-n` argument controls how many frames are saved.

Example:

```bash
python src/extract_frames.py --video data/videos/rory_005.mp4 --every-n 1
```

This creates extracted frames in the local `data/extracted_frames/` folder.

---

## Run Detection and Tracking

Before running the tracker, make sure that:

* the extracted frames exist
* the trained YOLO model weights exist
* the paths in `src/detect_video.py` are correct

Then run:

```bash
python src/detect_video.py
```

The script:

1. loads the YOLO model
2. reads the extracted video frames
3. detects possible golf ball candidates
4. tracks the ball using ROI logic and Kalman-based prediction
5. filters unlikely detections
6. saves detection and tracking results
7. saves annotated frames or visual output

Example output folder:

```text
output/detections/rory_005/
```

Generated files can include:

```text
detections.csv
tracking.csv
annotated_frames/
```

If the dataset or model weights are missing, the project will not run directly after cloning. The user must first add the required local files or adjust the paths in the script.

---

## Create Final Overlay Video

After running the detection script, the overlay script can be used to create the final visualisation:

```bash
python src/overlay.py
```

The overlay script reads the detection or tracking results and creates a visual trajectory overlay.

Example output:

```text
output/detections/rory_005/annotated_tracking.mp4
```

---

## How the Tracking Works

### 1. YOLO Detection

YOLO is used to detect possible golf ball candidates in each frame.

This works well when the ball is visible, but it can fail when the ball is very small, blurred or moving very fast.

---

### 2. ROI-Based Detection

Instead of searching the full frame all the time, the tracker uses a Region of Interest around the expected ball position.

This improves tracking because:

* the search area is smaller
* there are fewer false detections
* detection is focused on the likely ball location
* tracking after launch becomes more stable

---

### 3. Tracking States

The tracker works with different phases of the ball movement:

```text
tee
launch
flight
```

In the `tee` phase, the ball is still close to the starting position.

In the `launch` phase, the system detects when the ball leaves the tee.

In the `flight` phase, the tracker follows the moving ball using ROI-based detection and motion prediction.

---

### 4. Launch Detection

The system detects launch by checking whether the ball has moved far enough away from the tee or whether the movement between consecutive detected frames is large enough.

After launch is detected, the tee estimate is frozen. This prevents later detections from changing the original tee position.

---

### 5. Kalman-Based Motion Prediction

The tracker uses a Kalman-based approach to estimate the next ball position.

The state contains:

```text
x, y, vx, vy
```

where:

* `x` and `y` are the current ball position
* `vx` and `vy` are the estimated velocity values

The measurement contains:

```text
x, y
```

This helps the tracker continue following the ball when YOLO misses detections for a few frames.

---

### 6. Candidate Filtering

Detected candidates are filtered using different rules, for example:

* bounding box size
* aspect ratio
* circularity
* confidence score
* distance from the tee
* distance from the predicted position
* direction of motion
* plausibility of the movement

This reduces false positives and keeps the tracker focused on physically plausible ball movement.

---

## Training the Detector

The YOLO detector can be trained with:

```bash
python src/train_detector.py
```

The training script expects a YOLO dataset configuration file such as:

```text
data/raw/rucv_split/data.yaml
```

The training setup used for the project was based on YOLO and custom golf ball annotations.

Example YOLO training command:

```bash
yolo detect train model=yolo11n.pt data=data/raw/rucv_split/data.yaml epochs=15 imgsz=640 batch=8 device=0 project=models/experiments name=yolo11n_golf_ball
```

For small object detection, higher image resolution can help because the golf ball is very small in the frame.

---

## Dataset Preparation

The repository contains helper scripts for preparing dataset files.

Convert VOC annotations to YOLO format:

```bash
python src/convert_rucv_voc_to_yolo.py
```

Split the converted dataset into train, validation and test sets:

```bash
python src/split_rucv_dataset.py
```

The split script uses:

```text
80% train
10% validation
10% test
```

The prepared dataset is not included in the repository and must be added locally.

---

## Output

The project can generate:

* detection CSV files
* tracking CSV files
* annotated frames
* final overlay video

Example output structure:

```text
output/
└── detections/
    └── rory_005/
        ├── detections.csv
        ├── tracking.csv
        ├── annotated_frames/
        └── annotated_tracking.mp4
```

---

## Known Limitations

* The ball can be missed directly after impact because of motion blur.
* Very low-quality or low-frame-rate videos make tracking harder.
* The model depends strongly on the quality and amount of training data.
* False positives can still happen when small bright objects appear near the expected trajectory.
* The project currently requires local paths for the dataset and model weights.
* The repository does not include the dataset or trained model file.

---

## Possible Improvements

Possible future improvements include:

* add command-line arguments for video path, model path and output path
* include a small public demo video
* upload model weights separately using GitHub Releases
* improve launch detection
* tune the Kalman filter further
* train with more post-impact examples
* use higher-resolution training images
* improve false-positive filtering
* export the final video directly from the main pipeline

---

## Example Workflow

```bash
# 1. Clone repository
git clone https://github.com/KimBor04/golf-ball-tracker.git
cd golf-ball-tracker

# 2. Create virtual environment
python -m venv .venv

# 3. Activate environment on Windows PowerShell
.venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Add local input video
# Example: data/videos/example_video.mp4

# 6. Add local model weights
# Example: models/best.pt

# 7. Extract frames from the video
python src/extract_frames.py --video data/videos/example_video.mp4 --every-n 1

# 8. Check and adjust paths in src/detect_video.py

# 9. Run detection and tracking
python src/detect_video.py

# 10. Create final overlay video
python src/overlay.py
```

---

## Reproducibility Note

This repository contains the source code and documentation for the project.

It does not contain the complete dataset, videos, extracted frames or trained model weights. Therefore, the project is not fully runnable immediately after cloning. To reproduce the original demo, the required local data and model files must be added manually.

---

## Project Summary

This project tracks a golf ball in swing videos using YOLO detection, ROI-based tracking and Kalman-based motion prediction.

The main challenge is that the ball is very small and moves extremely fast after impact. A simple full-frame detector is therefore not enough. The project improves tracking by combining detection, ROI search windows, launch detection, motion prediction and filtering rules to create a more stable ball trajectory.

