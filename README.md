<div align="center">

<img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B35?style=for-the-badge"/>
<img src="https://img.shields.io/badge/OpenCV-4.13-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
<img src="https://img.shields.io/badge/Tests-35%20Passing-22C55E?style=for-the-badge"/>

# Intelligent Headlight System

**Real-time adaptive headlight control using computer vision.**  
Detects vehicles, people, and bikes — dims only the LED zones where they are,  
keeps full brightness everywhere else.

*Adaptive headlights exist in luxury cars costing ₹50L+.  
This project implements the core algorithm on a ₹2000 hardware budget.*

</div>

---

## The Problem

High-beam headlights cause glare for oncoming drivers and pedestrians — a leading cause of night-time road accidents. Traditional solutions are expensive, proprietary, and require custom hardware. 

This system solves it with a camera, a $5 microcontroller, and computer vision.

---

## How It Works

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Webcam    │────▶│  YOLOv8n   │────▶│ IOU Tracker │────▶│  mask_gen   │
│  Live Feed  │     │  Detector   │     │ Stable IDs  │     │ bbox → grid │
└─────────────┘     └─────────────┘     └─────────────┘     └──────┬──────┘
                                                                     │
                          ┌──────────────────────────────────────────┘
                          ▼
                   ┌─────────────┐     ┌─────────────┐
                   │  8×8 LED   │────▶│   Display   │
                   │    Mask     │     │  Simulation │
                   └─────────────┘     └─────────────┘
```

Each camera frame goes through five stages: capture → detect → track → map → render. The result is an 8×8 boolean grid where `True` means dim and `False` means full brightness — updated every frame in real time.

---

## The Math

### Grid Mapping  *(mask_gen.py)*

A bounding box `(x_min, y_min, x_max, y_max)` from a 1920×1080 frame maps to the 8×8 LED grid using:

```
Grid_col = ⌊ x × 8 / 1920 ⌋
Grid_row = ⌊ y × 8 / 1080 ⌋
```

Both corners of the bounding box are mapped independently. Every LED cell between them is dimmed. A ±1 cell margin is added on all sides to prevent edge glare.

### Distance Estimation  *(detector.py)*

Uses the pinhole camera model:

```
distance (m) = real_height (m) × focal_length (px)
               ─────────────────────────────────────
                      pixel_height (px)
```

Known heights: person = 1.75m, car = 1.50m, truck = 2.80m.  
Accuracy after calibration: **±20%** — sufficient for LED zone selection.

### IOU Matching  *(tracker.py)*

```
        Intersection Area
IOU =  ─────────────────
           Union Area
```

Tracks with IOU > 0.3 across consecutive frames are considered the same object. Unmatched tracks survive up to 5 frames before being dropped — eliminating LED flicker from single-frame detection misses.

---

## Project Structure

```
intelligent-headlight-system/
│
├── src/
│   ├── detector.py          YOLOv8 inference + distance estimation
│   ├── tracker.py           IOU-based object tracker (stable IDs)
│   ├── mask_gen.py          Bounding box → 8×8 LED grid mapping
│   ├── led_display.py       Simulated LED matrix (OpenCV + numpy)
│   ├── main.py              Pipeline entry point
│   └── detector_test.py     Quick visual sanity check (webcam only)
│
├── calibration/
│   ├── focal_calibration.py One-time focal length calibration script
│   └── calibration.json     Generated after calibration (Week 3)
│
├── tests/
│   ├── test_mask_gen.py     25 tests — geometry and grid mapping
│   ├── test_detector.py     5 tests  — distance formula
│   └── test_led_display.py  5 tests  — LED rendering
│
├── assets/                  Sample videos for testing without webcam
├── demo/                    Screenshots and demo recordings
└── requirements.txt
```

---

## Quick Start

```bash
# Clone and set up
git clone https://github.com/your-repo/intelligent-headlight-system
cd intelligent-headlight-system
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Verify everything works — should show 35 passed
pytest tests/ -v

# Sanity check detector alone first
python src/detector_test.py

# Run full pipeline — live webcam
python src/main.py

# Run on a video file
python src/main.py --source assets/sample.mp4

# Run without webcam (synthetic detections)
python src/main.py --stub
```

---

## Calibration

Run once before using the system for the first time:

```bash
python calibration/focal_calibration.py
```

Stand exactly **2 metres** from the webcam. Press `SPACE` three times when your full body is visible in the frame. Copy the printed `FOCAL_LENGTH` value into `src/detector.py`:

```python
FOCAL_LENGTH = YOUR_VALUE_HERE   # line 12 of detector.py
```

---

## Detection Classes and Colours

| Class | Box Colour | Real Height Used |
|-------|-----------|-----------------|
| Person | Blue | 1.75 m |
| Car | Green | 1.50 m |
| Truck | Orange | 2.80 m |
| Bus | Purple | 3.20 m |
| Motorcycle | Yellow-green | 1.20 m |
| Bicycle | Pink | 1.10 m |
| *Coasting (any)* | *Grey* | *— tracker holding* |

Boxes drawn in their class colour when actively detected. Grey when the tracker is holding from previous frames.  
Boxes drawn **3px thick** when the object is within 5 metres. **2px** otherwise.

---

## LED Mask Convention

| Value | LED State | When |
|-------|-----------|------|
| `True` | **OFF** — dimmed | Object detected in this zone |
| `False` | **ON** — full brightness | Clear road ahead |

---

## What We Have Built

### ✅ Completed

**Geometry module (`mask_gen.py`)** — fully implemented and battle-tested.
Converts any bounding box to LED grid coordinates using the linear mapping formula. Handles all edge cases: coordinates at frame boundary, zero-area boxes, inverted boxes, objects spanning multiple LED zones, margin clamping at grid edges. 25 passing unit tests covering every case.

**Real-time detection (`detector.py`)** — YOLOv8n running live.
Filters to 6 road-relevant classes. Applies a horizon filter (ignores detections above 55% frame height — eliminates false triggers from signs and bridges). Estimates distance using the pinhole model. Falls back to synthetic detections gracefully when no camera is available.

**IOU Tracker (`tracker.py`)** — stable object identities across frames.
Greedy IOU matching assigns consistent IDs to the same object across frames. Coasting for up to 5 missed frames prevents LED flicker. No external library required — pure numpy.

**LED Simulation (`led_display.py`)** — real-time visual display.
Renders the 8×8 boolean mask as a 534×534 pixel panel using pure numpy array slicing. Yellow-white for ON cells, dark grey for OFF cells, inner glow effect. Platform-independent — no dependency on specific OpenCV installation variants.

**Full pipeline (`main.py`)** — all modules wired together.
Side-by-side window: annotated camera feed on the left, LED matrix on the right. Per-class colour coding, FPS counter, object count, coasting detection visualisation. Supports webcam, video file, and headless stub mode.

**Test suite** — 35 tests, all passing.
Covers the grid mapping formula, edge cases, distance calculation, LED rendering correctness, and cross-module integration. Runs in under 1 second.

---

## What We Are Building Next

### 🔨 In Progress

**Oncoming vehicle detection**  
Same-direction vehicles (driving away) should not trigger dimming — only oncoming vehicles cause glare. Implementing bbox growth tracking inside the IOU tracker: a bounding box that grows larger across frames indicates an approaching vehicle. Oncoming label added to track dict; `mask_gen` filters by direction before generating the mask.

**Distance-based dimming zones**  
Currently all detected objects trigger full dimming. The upgrade introduces three zones:
```
> 40m   →   no action
15–40m  →   40% dim
< 15m   →   full dim
```
`generate_mask` returns a float array (0.0–1.0) instead of boolean, and `led_display` scales LED brightness accordingly.

### 📋 Planned

**Physical LED matrix**  
WS2812B 8×8 matrix connected to Arduino Uno via USB serial. `serial_writer.py` (currently empty) sends the mask over serial every frame. Arduino receives and sets individual LED brightness. This is the single biggest demo impact — a physical panel responding to a live camera.

**Night detection (`night_detector.py`)**  
YOLOv8 was trained on daytime images and struggles at night when only headlights are visible. A parallel classical CV pipeline: brightness threshold → connected components → blob pairing by separation width (narrow pair = motorcycle, medium = car, wide = truck). Runs alongside YOLO, merges detections, activated automatically when mean frame brightness drops below 80/255.

**Fine-tuned night model**  
YOLOv8n fine-tuned on BDD100K night sequences using Google Colab (T4 GPU, ~50 epochs, freeze=10 to prevent catastrophic forgetting of daytime performance). Replaces the classical night detector for better accuracy in complex night scenes.

**Focal calibration JSON**  
Non-linear zone mapping using `calibration.json` produced by the calibration script. Replaces the current linear formula for more accurate LED zone selection when the camera has significant lens distortion.

---

## Team

| Person | Role |
|--------|------|
| Sri | Geometry module (`mask_gen.py`), math documentation, integration |
| Person A | Detection (`detector.py`), tracking (`tracker.py`), night model |
| Person B | LED display, hardware (`serial_writer.py`), Arduino integration |

---

## Results

| Metric | Value |
|--------|-------|
| Detection classes | 6 (person, car, truck, bus, motorcycle, bicycle) |
| Grid resolution | 8 × 8 (64 zones) |
| Pipeline FPS (CPU) | 10–15 FPS on YOLOv8n |
| Distance accuracy | ±20% after focal calibration |
| Test coverage | 35 tests, 100% passing |
| Distance range | Reliable 2m – 30m |

---

## Requirements

```
ultralytics >= 8.0.0
opencv-python >= 4.8.0
numpy >= 1.24.0
pytest >= 7.4.0
pyserial >= 3.5        # Week 2: Arduino serial
```

Python 3.10 or higher. No GPU required — runs on CPU with YOLOv8n.

---

<div align="center">

*Built from scratch. Every formula derived. Every edge case tested.*

</div>