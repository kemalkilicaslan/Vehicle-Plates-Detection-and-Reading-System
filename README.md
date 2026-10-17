# Vehicle Plates Detection and Reading System

## 1. Introduction

This project implements a real-time vehicle plate detection and reading system that combines a custom-trained RF-DETR object detector with an ONNX-based vehicle plate OCR engine. The system processes traffic footage to detect vehicle plates, reads the plate characters, formats and validates them against the Turkish plate standard, and overlays the recognized plate on an annotated output video.

Automatic vehicle plate recognition (AVPR) is used in intelligent transportation systems, traffic enforcement, parking management, and access control. Pairing a transformer-based detector (RF-DETR) with a fast, hardware-accelerated OCR model, the system reads plates and runs on both Apple Silicon (CoreML) and NVIDIA GPUs (CUDA).

The pipeline runs end to end: it detects plates in every frame, crops and reads each plate, formats and validates the text against the Turkish plate standard, filters out uncertain reads with a confidence check, and draws the recognized plate onto the output video.

**Core Features:**
- Real-time plate detection using a custom RF-DETR Large model
- Vehicle plate reading with Fast Plate OCR (ONNX, CoreML/CPU accelerated)
- Turkish plate formatting and validation with character correction
- Confidence-based filtering to skip uncertain / unreliable plate reads
- IoU-based plate tracking that locks in the first reading exceeding the confidence threshold and keeps it constant across every frame the plate appears in
- Fixed-size plate label block with auto-scaled text for consistent, readable overlays
- Live preview via `cv2.imshow` with automatic headless fallback
- Video output with annotated plates
- Reproducible RF-DETR Large training on a custom Turkish plate dataset via the included Google Colab notebook

## 2. Methodology / Approach

The system runs a single RF-DETR detector per frame to locate plates, then crops each detected plate and reads it with an OCR model. Detected plates are matched to persistent tracks by box overlap so each plate's first reading that exceeds the confidence threshold is locked in and reused unchanged on later frames. Readings are corrected, formatted, and validated against the Turkish plate format before being drawn onto the frame.

### 2.1 System Architecture

The vehicle plate detection and reading pipeline consists of:

1. **Detection:** RF-DETR Large detects plates (class `1`) in each frame
2. **Class Filtering:** Only plate detections are kept
3. **Plate Tracking:** Each detection is matched to an existing track by IoU overlap so the same plate keeps its reading; unmatched detections start new tracks and unseen tracks are aged out after a few frames
4. **Plate OCR:** A plate box is cropped and read by Fast Plate OCR only while its track has no locked-in text yet
5. **Text Formatting:** Readings are corrected, formatted, and validated against the Turkish plate format
6. **Confidence Filtering:** The first reading whose confidence exceeds the threshold is locked into the track; readings below the threshold are discarded and the plate keeps being read until one qualifies
7. **Visualization:** The locked-in plate text is overlaid, staying constant across every frame the plate appears in

### 2.2 Implementation Strategy

Detection uses the RF-DETR framework with custom pretrained weights (`checkpoint_best_total.pth`), run at a fixed input resolution (1280×720) and rescaled back to the original frame size. Plate crops are passed to the `cct-s-v2-global-model` OCR model through the ONNX runtime, using the CoreML execution provider on Apple Silicon and falling back to CPU when unavailable. A per-character confidence check drops uncertain reads so a wrong plate is skipped rather than drawn. To keep readings stable, each detection is matched to a persistent track by Intersection-over-Union (IoU): the first reading whose confidence exceeds the threshold is locked into the track and reused unchanged on every later frame it appears in, so the displayed text stays constant, and tracks survive brief detection gaps for a configurable number of frames. Each accepted reading is normalized and validated against the Turkish plate structure, then rendered inside a fixed-size white label block whose text is auto-scaled to fit.

## 3. Mathematical Framework

### 3.1 Detection Rescaling

RF-DETR runs on a fixed input resolution, so detections are scaled back to the original frame size:

$$x_{\text{orig}} = x_{\text{det}} \cdot \frac{W_{\text{orig}}}{W_{\text{input}}}, \qquad y_{\text{orig}} = y_{\text{det}} \cdot \frac{H_{\text{orig}}}{H_{\text{input}}}$$

where $(W_{\text{input}}, H_{\text{input}}) = (1280, 720)$ and $(W_{\text{orig}}, H_{\text{orig}})$ is the source frame size. Coordinates are clipped to the frame bounds.

### 3.2 Reading Confidence

Each plate reading is accepted only if its confidence meets the threshold. Confidence is the lowest per-character probability over the real (non-padding) characters:

$$c = \min_{i \,:\, ch_i \neq \text{pad}} p_i$$

where $p_i$ is the predicted probability of character $i$. A reading is locked into its track and drawn only when $c \geq$ `OCR_CONFIDENCE_THRESHOLD` (default `0.95`); once locked it is reused unchanged on every later frame of that plate, while uncertain reads are skipped instead of shown.

### 3.3 Plate Tracking (IoU)

Each detected plate box is matched to an existing track using Intersection-over-Union:

$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

A detection is treated as the same plate as an existing track when its best overlap satisfies $\text{IoU} \geq$ `PLATE_MATCH_IOU` (default `0.2`), in which case the track's stored text is reused. Tracks that go unmatched are kept for up to `PLATE_MAX_MISSES` frames (default `60`) before being dropped, bridging brief detection gaps.

## 4. Requirements

`requirements.txt`

```txt
opencv-python>=4.8.0
numpy>=1.24.0
supervision>=0.16.0
rfdetr>=1.4.0
fast-plate-ocr>=0.3.0
roboflow>=1.1.0
tqdm>=4.62.0
pillow>=8.0.0
```

## 5. Installation & Configuration

### 5.1 Environment Setup

```bash
# Clone the repository
git clone https://github.com/kemalkilicaslan/Vehicle-Plates-Detection-and-Reading-System.git
cd Vehicle-Plates-Detection-and-Reading-System

# Install required packages
pip install -r requirements.txt
```

### 5.2 Project Structure

```
Vehicle-Plates-Detection-and-Reading-System
├── RF-DETR-Custom-Vehicle-Plates-Detection-Training.ipynb   # RF-DETR training notebook (Google Colab)
├── Vehicle-Plates-Detection-and-Reading-System.py          # detection + reading script
├── config.py                                               # tunable settings (paths, device, OCR, thresholds, tracking)
├── checkpoint_best_total.pth                               # custom-trained RF-DETR weights
├── Vehicle-Plates.mp4                                       # input video
├── Vehicle-Plates-Detection-and-Reading.mp4                # annotated output video
├── Dockerfile
├── .dockerignore
├── docker-compose.yml
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
```

### 5.3 Path Configuration

All adjustable parameters (sections 5.3-5.7 below) live in `config.py`; update them there without touching the main script. Set the model, input, and output file names in `config.py`. The main script resolves them relative to its own folder (`BASE_DIR`), so the program runs the same no matter which directory it is launched from:

```python
# config.py
MODEL_WEIGHTS = "checkpoint_best_total.pth"                  # RF-DETR weights
INPUT_VIDEO   = "Vehicle-Plates.mp4"                         # input video
OUTPUT_VIDEO  = "Vehicle-Plates-Detection-and-Reading.mp4"  # output video
```

### 5.4 Detection & Device Parameters

```python
# Compute device for RF-DETR
#   "mps" → Apple Silicon GPU | "cuda" → NVIDIA GPU | "cpu" → fallback
DEVICE = "mps"

# Frame size each frame is resized to before RF-DETR detection
RFDETR_INPUT_SIZE = (1280, 720)   # (width, height)

# Class IDs — adjust to match your dataset's label order
PLATE_CLASS_IDS     = [1]   # plates → OCR
DETECTION_THRESHOLD = 0.3
```

### 5.5 OCR Parameters

```python
# Models: cct-s-v2-global-model (accurate) | cct-xs-v2-global-model (fastest)
OCR_MODEL_NAME = "cct-s-v2-global-model"
OCR_PROVIDERS  = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

# Minimum OCR confidence (0.0–1.0) to accept and lock in a plate reading
OCR_CONFIDENCE_THRESHOLD = 0.95
```

### 5.6 Tracking Parameters

```python
# Minimum box overlap (IoU) to treat a plate as the same one seen before,
# so its first read text is reused instead of read again
PLATE_MATCH_IOU = 0.2

# How many consecutive frames a plate may stay undetected before its track
# (and its locked-in text) is dropped — bridges brief detection gaps
PLATE_MAX_MISSES = 60
```

### 5.7 Label & Display Configuration

```python
# Fixed size (in pixels) of the white label block that shows the plate reading
LABEL_BOX_WIDTH  = 180
LABEL_BOX_HEIGHT = 40

# Display window scale (1.0 = original, 0.5 = half size)
DISPLAY_SCALE = 0.5
```

## 6. Usage / How to Run

### 6.1 Basic Execution

```bash
python Vehicle-Plates-Detection-and-Reading-System.py
```

**Requirements:**
- Input video: set via `video_path` (place in project directory)
- RF-DETR weights: `checkpoint_best_total.pth` (place in project directory)

**Live Window:**
- A resizable preview window titled **"Vehicle Plates Detection and Reading System"** opens during playback (scaled by `DISPLAY_SCALE`). If no GUI is available, the system runs headless and still saves the output video.

**Controls:**
- Press `q` to quit during playback
- Output saved to: path set in `output_file`

### 6.2 Customizing Input Video

Change the video filename in the script:

```python
video_path = str(BASE_DIR / "Vehicle-Plates.mp4")
```

### 6.3 Tuning Detection & OCR

```python
# Detection sensitivity
DETECTION_THRESHOLD = 0.3   # more detections
DETECTION_THRESHOLD = 0.6   # fewer false positives

# OCR reliability
OCR_CONFIDENCE_THRESHOLD = 0.95   # fewer but more reliable reads
OCR_CONFIDENCE_THRESHOLD = 0.6    # more reads

# Faster OCR on limited hardware
OCR_MODEL_NAME = "cct-xs-v2-global-model"
```

### 6.4 Training the Custom Model (Notebook)

The `checkpoint_best_total.pth` weights used by the script are produced by the included Google Colab notebook `RF-DETR-Custom-Vehicle-Plates-Detection-Training.ipynb`. Open it in Colab with a GPU runtime and run the cells in order.

```bash
# Notebook dependencies (installed in the first cell)
pip install -q rfdetr>=1.4.0 supervision roboflow
```

The notebook workflow:

1. **Install & import** the RF-DETR, Supervision, and Roboflow libraries
2. **Download the dataset** from Roboflow in COCO format (`vehicle-plates-in-tr`, version 10) using a `ROBOFLOW_API_KEY` stored in Colab `userdata`
3. **Train** `RFDETRLarge` for `50` epochs (`batch_size=4`, `grad_accum_steps=4`)
4. **Visualize** the training/validation losses and success metrics (`metrics_plot.png`)
5. **Load the best weights**, optimize for inference, and print the model architecture
6. **Evaluate** the mean Average Precision (mAP) on the test set with `supervision.metrics.MeanAveragePrecision`
7. **Visualize** ground-truth annotations vs. RF-DETR detections on the test images
8. **Export** the training outputs (weights + metrics) as `output.zip` for download

The trained `checkpoint_best_total.pth` from `/content/output` is then placed next to the script for detection and reading.

## 7. Docker Deployment

The system can be packaged and run in a container so that the code, its
dependencies and the runtime environment stay consistent across machines.

### Files

- `Dockerfile` - builds the image (Python 3.11 + OpenCV/ffmpeg system libraries + Python dependencies).
- `.dockerignore` - keeps large media and model files out of the build context.
- `docker-compose.yml` - runs the container and mounts the project folder for input/output.

### Build and run

```bash
docker compose up --build

# or with the Docker CLI directly
docker build -t vehicle-plates-detection-reading:latest .
docker run --rm -v "$(pwd)":/app vehicle-plates-detection-reading:latest
```

### Notes

- Input and output files are read from and written to the project folder through
  the bind mount, so results appear on the host after the run.
- The bundled model checkpoint (`checkpoint_best_total.pth`) is used through the
  bind mount. Additional model artifacts (RF-DETR, OCR) are downloaded on first
  run and cached in the `torch-cache` volume.
- The script calls `cv2.imshow`, which needs a graphical display. On a headless
  server the output video is still written to disk; the live preview window is
  only available on Linux with X11 forwarding (see the commented lines in
  `docker-compose.yml`).

## 8. Application / Results

### 8.1 Input Video

**Vehicle Plates:**

[![Vehicle Plates](https://github.com/kemalkilicaslan/Vehicle-Plates-Detection-and-Reading-System/blob/main/Vehicle-Plates.webp)](https://www.youtube.com/watch?v=pIjVMnBeob4)

### 8.2 Output Video

**Vehicle Plates Detection and Reading:**

[![Vehicle Plates Detection and Reading](https://github.com/kemalkilicaslan/Vehicle-Plates-Detection-and-Reading-System/blob/main/Vehicle-Plates-Detection-and-Reading.webp)](https://www.youtube.com/watch?v=XBEEvqt3XlE)

### 8.3 Detection & Reading Pipeline

For each frame the system:

1. Detects plates with RF-DETR
2. Matches each detection to an existing track by IoU, reusing the plate's locked-in reading when found
3. Crops and reads a plate with Fast Plate OCR only while its track has no locked-in text yet
4. Locks in the first reading whose confidence exceeds the threshold and discards reads below it
5. Corrects, formats, and validates the reading against the Turkish plate format
6. Draws the recognized plate (black text on a fixed-size white label above a red plate box), kept constant across every frame of that plate

### 8.4 Turkish Plate Formatting

Readings are normalized to the Turkish plate structure `NN L(1–3) N(1–4)`:

| Segment | Content | Length | Example |
|---------|---------|--------|---------|
| Province code | Digits | 2 | `06` |
| Letter group | Letters (`X`, `Q`, `W` removed) | 1–3 | `ABC` |
| Number group | Digits | 1–4 | `1234` |

Common OCR confusions are corrected per segment (e.g. `O↔0`, `I↔1`, `S↔5`, `G↔6`, `J↔3`, `A↔4`), and readings that do not match the format are rejected.

### 8.5 System Parameters

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| Detection Threshold | 0.3 | - | RF-DETR confidence threshold |
| RF-DETR Input Size | 1280×720 | px | Detector input resolution |
| Plate Class ID | 1 | - | Class used for OCR |
| OCR Confidence Threshold | 0.95 | - | Minimum confidence to accept and lock a plate reading |
| Plate Match IoU | 0.2 | - | Min overlap to reuse a plate's track |
| Plate Max Misses | 60 | frames | Frames a track survives without a match |
| Label Box Size | 180×40 | px | Fixed white label block for the reading |
| Display Scale | 0.5 | - | Live preview window scale |

## 9. Tech Stack

### 9.1 Core Technologies

- **Programming Language:** Python 3.8+
- **Computer Vision:** OpenCV 4.8+
- **Object Detection:** RF-DETR Large (custom weights)
- **Plate OCR:** Fast Plate OCR (ONNX runtime, CoreML/CPU)
- **Video Processing:** Supervision VideoInfo + OpenCV VideoWriter

### 9.2 Libraries & Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| opencv-python | 4.8+ | Video I/O, image cropping, rendering |
| numpy | 1.24+ | Array operations, coordinate calculations |
| supervision | 0.16+ | Detections container, video info, frame generator, metrics |
| rfdetr | 1.4+ | RF-DETR model training, inference and detection |
| fast-plate-ocr | 0.3+ | Vehicle plate character recognition (OCR) |
| roboflow | 1.1+ | Custom dataset download for training (notebook) |
| tqdm | 4.62+ | Progress bars during training/evaluation (notebook) |
| pillow | 8.0+ | Image loading for detection and visualization (notebook) |

### 9.3 Model Architecture

**RF-DETR Large:**
- **Weights:** `checkpoint_best_total.pth` (custom-trained)
- **Input Resolution:** 1280×720 pixels
- **Architecture:** Transformer-based real-time detection (DETR family)
- **Detection Class:** plate (`1`)
- **Compute Device:** MPS (Apple Silicon) / CUDA (NVIDIA) / CPU

**Fast Plate OCR:**
- **Model:** `cct-s-v2-global-model` (accurate) or `cct-xs-v2-global-model` (fastest)
- **Runtime:** ONNX with CoreML acceleration on Apple Silicon, CPU fallback
- **Output:** Per-character predictions with confidence probabilities

### 9.4 Pipeline Components

| Component | Type | Purpose |
|-----------|------|---------|
| RF-DETR Large | Detector | Plate detection |
| Fast Plate OCR | Recognizer | Plate character reading |
| box_iou | Tracker | IoU matching to reuse a plate's locked-in reading |
| format_plate_text | Formatter | Turkish plate normalization and validation |
| reading_confidence | Filter | Per-character confidence gating |
| annotate_plate | Visualizer | Draw plate box and fixed-size label |

## 10. License

This project is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)](https://github.com/kemalkilicaslan/Vehicle-Plates-Detection-and-Reading-System/blob/main/LICENSE).

## 11. References

1. Roboflow [RF-DETR](https://github.com/roboflow/rf-detr) Documentation.
2. [Fast Plate OCR](https://github.com/ankandrew/fast-plate-ocr) Documentation.
3. Roboflow [Supervision](https://supervision.roboflow.com/latest/) Documentation.
4. OpenCV [Drawing Functions](https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html) Documentation.
5. Roboflow Universe [Vehicle Plates in TR](https://universe.roboflow.com/vehicle-plate-detection-6l2v6/vehicle-plates-in-tr) Dataset.

---

### Acknowledgments

This project uses RF-DETR from Roboflow for plate detection, the Fast Plate OCR library for vehicle plate reading, and the Supervision library for video handling. Thanks to the computer vision community for their open-source tools for traffic analysis and AVPR.

---

**Note:** This system is intended for research, education, and traffic analysis purposes. For legal enforcement or identity applications, ensure compliance with local regulations and data-protection laws. Plate readings may vary based on camera angle, plate condition, and environmental conditions.
