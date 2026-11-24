"""Train seg YOLOv8-seg, then evaluate on the test split."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.runner import main
from detectors.yolov8_seg.build import build_model

if __name__ == "__main__":
    main("train", task="seg", version="yolov8_seg", build_model=build_model)
