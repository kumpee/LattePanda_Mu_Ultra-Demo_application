# Demo Application for LattePanda Mu Ultra (Intel Core Ultra 7): Real-Time Webcam Video Adjustment Using Hand Gestures

This Python demo was prepared for my review of the **LattePanda Mu Ultra**, powered by an **Intel Core Ultra 7 256V** processor with **16 GB RAM**, running **Ubuntu 26.04 LTS**. The review is available on [CNX Software](https://www.cnx-software.com/).

The application combines **YOLO26, OpenVINO, MediaPipe, and OpenCV** to demonstrate several computer vision tasks using a USB webcam.

## Features

- Real-time depth estimation using a YOLO26 depth model, displayed as a color overlay
- YOLO26 object detection with bounding boxes, class labels, and confidence scores
- MediaPipe face, hand, and body pose landmark tracking
- Gesture-based image adjustment of hue, saturation, contrast, and brightness
- Independent CPU, GPU, or NPU inference selection for YOLO26 object detection and depth estimation through OpenVINO
- Two display windows: the adjusted webcam image and a second view with depth, detections, and landmarks

The application tracks up to two hands. The **left hand** adjusts hue and saturation, while the **right hand** adjusts contrast and brightness. Pinch your thumb and index finger together to activate an adjustment, move your hand to change the values, and release the pinch to keep the settings.

## Requirements

- LattePanda Mu Ultra or another system compatible with the required software
- Ubuntu 26.04 LTS (review configuration)
- Python 3 with `venv` support
- USB webcam
- OpenVINO-compatible hardware and drivers for GPU/NPU inference (when used)
- Internet connection on the first run if model files have not already been downloaded

## Setup

Create and activate a Python virtual environment:

```bash
python3 -m venv venv_demo_hand_full
source venv_demo_hand_full/bin/activate
python -m pip install --upgrade pip
```

Install the required Python packages:

```bash
pip install numpy opencv-python mediapipe
pip install "ultralytics[export-openvino]"
pip install "openvino==2026.2.0"
```

## Running the Demo

Connect a USB webcam and run:

```bash
python demo_hand_009.py
```

By default, the script uses **`yolo26s`** for object detection and **`yolo26n-depth`** for depth estimation, with both inference tasks initially set to **CPU**. Missing official YOLO26 checkpoints can be downloaded automatically, and `.pt` models are exported to OpenVINO FP16 for inference. The MediaPipe hand, face, and pose models are also downloaded automatically if missing.

For example, to start both YOLO26 tasks on the GPU:

```bash
python demo_hand_009.py --yolo-device GPU --device GPU
```

To run detection on the GPU and depth estimation on the NPU:

```bash
python demo_hand_009.py --yolo-device GPU --device NPU
```

To disable either task at startup:

```bash
python demo_hand_009.py --no-depth
python demo_hand_009.py --no-objects
```

Show the command-line options with:

```bash
python demo_hand_009.py --help
```

## Keyboard Controls

| Key | Action |
| --- | --- |
| `C` / `G` / `N` | Select CPU / GPU / NPU for object detection (lowercase key input) |
| `Shift+C` / `Shift+G` / `Shift+N` | Select CPU / GPU / NPU for depth estimation |
| `Space` | Switch webcam resolution |
| `R` | Reset image adjustments |
| `S` | Save the annotated output image |
| `Q` or `Esc` | Quit |

**Code note:** The supplied `demo_hand_009.py` has a small issue in the `D` and `O` toggle handlers: they refer to undefined `depth` and `yolo` variables. Update those conditions to use `depth_model` and `det_model`, respectively, before using these keys to toggle the two overlays.

## Notes

This is an experimental demonstration prepared for a hardware review, not a production application. The left display shows the adjusted webcam image. The second display adds YOLO26 depth visualization, object detections, and MediaPipe landmarks. Actual frame rates depend on the inference devices, model sizes, camera resolution, and processing overhead.

## Review

More information about the hardware and benchmarks is available at [CNX Software](https://www.cnx-software.com/).
