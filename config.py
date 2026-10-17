# Vehicle Plates Detection and Reading System — Configuration

# Model weights and video file names
MODEL_WEIGHTS = "checkpoint_best_total.pth"   # RF-DETR weights
INPUT_VIDEO   = "Vehicle-Plates.mp4"          # input video file name
OUTPUT_VIDEO  = "Vehicle-Plates-Detection-and-Reading.mp4"  # output video file name

# Compute device for RF-DETR ("mps" = Apple Silicon GPU, "cuda" = NVIDIA GPU, "cpu" = fallback)
DEVICE = "mps"

# Fast Plate OCR settings (cct-s-v2-global-model = accurate, cct-xs-v2-global-model = fastest)
OCR_MODEL_NAME = "cct-s-v2-global-model"
OCR_PROVIDERS = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

# Minimum OCR confidence to accept a plate reading
OCR_CONFIDENCE_THRESHOLD = 0.95

# Frame size (width, height) each frame is resized to before RF-DETR detection
RFDETR_INPUT_SIZE = (1280, 720)

# Class ID(s) for plates and detection confidence threshold
PLATE_CLASS_IDS = [1]
DETECTION_THRESHOLD = 0.3

# Display window scale (1.0 = original, 0.5 = half size)
DISPLAY_SCALE = 0.5

# Fixed size (in pixels) of the white label block that shows the plate reading
LABEL_BOX_WIDTH = 180
LABEL_BOX_HEIGHT = 40

# Minimum box overlap (IoU) to treat a plate as the same one seen before
PLATE_MATCH_IOU = 0.2

# Frames a plate may stay undetected before its track (and locked text) is dropped
PLATE_MAX_MISSES = 60

# Turkish plate character correction maps
CHAR_TO_INT = {"O": "0", "I": "1", "J": "3", "A": "4", "G": "6", "S": "5"}
INT_TO_CHAR = {"0": "O", "1": "I", "3": "J", "4": "A", "6": "G", "5": "S"}
