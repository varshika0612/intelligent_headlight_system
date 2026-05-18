# Intelligent Matrix Headlight System
### Week 1 — Detection Pipeline & Simulated LED Matrix

> **Hackathon track** | Team of 2 | Status: ✅ Week 1 Complete

---

## What this does

A real-time computer vision pipeline that:
1. Captures live webcam feed
2. Detects vehicles, people, and bikes using YOLOv8
3. Estimates distance to each detected object
4. Renders a simulated 8×8 LED matrix that dims in zones where objects are detected

No hardware required to run Week 1 — everything runs on a laptop with a webcam.

---

## Demo

```
[ Camera Feed + Bounding Boxes ]  |  [ Simulated LED Matrix ]
  ID:0 person 2.3m                |   ■ ■ ■ □ □ ■ ■ ■
  ID:1 car 18.4m                  |   ■ ■ ■ □ □ ■ ■ ■
                                  |   ■ ■ ■ ■ ■ ■ ■ ■
                                  |   ■ ■ ■ ■ ■ ■ ■ ■
  ■ = LED ON   □ = LED OFF (glare zone)
```

---

## Project Structure

```
intelligent-headlight-system/
│
├── src/
│   ├── detector.py          # YOLOv8 inference + distance estimation
│   ├── tracker.py           # IOU-based object tracker (stable IDs)
│   ├── mask_gen.py          # Maps bounding boxes → LED grid indices
│   ├── led_display.py       # Simulated 8×8 LED matrix (OpenCV)
│   └── main.py              # Entry point — wires all modules together
│
├── calibration/
│   ├── focal_calibration.py # One-time script to calibrate focal length
│   └── calibration.json     # Generated in Week 3 (hardware calibration)
│
├── tests/
│   ├── test_detector.py
│   ├── test_mask_gen.py
│   └── test_led_display.py
│
├── demo/                    # Store demo screenshots and videos here
├── assets/                  # Sample videos for testing without webcam
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Requirements

- Python 3.10+
- Webcam
- No GPU needed — runs on CPU

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/intelligent-headlight-system.git
cd intelligent-headlight-system

# 2. (Recommended) create a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

`requirements.txt`:
```
ultralytics
opencv-python
numpy
pyserial
filterpy
```

> YOLOv8n model weights (~6 MB) download automatically on first run. No manual download needed.

---

## Running

### Main pipeline (camera + LED display)
```bash
cd src
python main.py
```

Press `Q` to quit.

### Test detector only
```bash
cd src
python detector.py
```

### Calibrate focal length (do this once)
```bash
cd calibration
python focal_calibration.py
```

Stand exactly **2 metres** from your webcam, press **SPACE**.  
Copy the printed `FOCAL_LENGTH` value into `src/detector.py`.

---

## Module Breakdown

### `detector.py` — YOLOv8 Inference
Wraps YOLOv8n to detect road users in a single frame.

**Detects:**
| Class | Label |
|---|---|
| 0 | person |
| 1 | bicycle |
| 2 | car |
| 3 | motorcycle |
| 5 | bus |
| 7 | truck |

**Key settings:**
```python
CONFIDENCE_THRESHOLD = 0.45   # detections below this are ignored
HORIZON_RATIO = 0.55          # ignore detections in top 55% of frame (sky)
FOCAL_LENGTH = 347.82         # calibrate this to your webcam (see below)
```

**Distance estimation:**  
Uses bounding box height + known real-world object heights to estimate distance in metres.
```
distance = (real_height × focal_length) / bbox_pixel_height
```
Smoothed over last 5 frames to prevent flickering.

---

### `tracker.py` — IOU Object Tracker
Assigns stable IDs to detections across frames using Intersection over Union (IOU) matching.

- Prevents LED flicker when a detection is missed for 1–2 frames
- Tracks up to `max_lost_frames=5` frames of absence before dropping a track
- No external tracking library needed — pure numpy

---

### `mask_gen.py` — Bounding Box → LED Grid Mapping
Converts tracked object bounding boxes into an 8×8 boolean LED mask.

- `True` = LED OFF (object in this zone — preventing glare)
- `False` = LED ON (clear road — full illumination)
- Adds ±1 LED margin padding around each bounding box
- Uses linear mapping in Week 1 — replaced with calibration JSON in Week 3

---

### `led_display.py` — Simulated LED Matrix
Renders an 8×8 LED grid using OpenCV as a visual stand-in for physical hardware.

- Bright yellow-white = LED ON
- Dark grey = LED OFF
- Hackathon demo fallback if physical hardware isn't at venue

---

### `main.py` — Pipeline Entry Point
Wires all modules into a single loop:

```
webcam → detector → tracker → mask_gen → led_display
                                       → serial_writer (Week 2)
```

Displays a side-by-side window: camera feed on left, LED matrix on right.

---

## Calibrating Focal Length

This step is required for accurate distance estimation.

1. Run `calibration/focal_calibration.py`
2. Stand exactly **2 metres** from the webcam
3. Press **SPACE** when your bounding box is visible
4. Copy the printed focal length into `src/detector.py`:
```python
FOCAL_LENGTH = YOUR_VALUE_HERE
```

Expected distance accuracy after calibration: **±20%**  
Good enough for LED zone selection — this is not a precision depth sensor.

---

## Known Limitations (Week 1)

| Limitation | Fix planned |
|---|---|
| LED mapping is linear, not calibrated to physical hardware | Week 3 — calibration JSON |
| No serial comms to MCU yet | Week 2 — `serial_writer.py` |
| No agentic persistence (LEDs restore instantly) | Week 2 — `agent.py` |
| Distance accuracy ±20% | Acceptable for prototype |
| Low light performance depends on webcam quality | Test with a decent webcam |

---

## Week-by-Week Progress

| Week | Status | Focus |
|---|---|---|
| **Week 1** | ✅ Complete | Detector, tracker, mask gen, sim LED display |
| **Week 2** | 🔄 In progress | Agentic controller, serial writer, hardware handshake |
| **Week 3** | ⏳ Upcoming | Live pipeline, hardware calibration, end-to-end test |
| **Week 4** | ⏳ Upcoming | Polish, demo video, code freeze, pitch prep |

---

## Team

| Person | Owns |
|---|---|
| Person A | `detector.py`, `tracker.py`, `agent.py` |
| Person B | `led_display.py`, `mask_gen.py`, `serial_writer.py` |
| Both | `main.py` — coordinate before editing |

---

## Branch Strategy

```
main   ← clean, demo-ready code (merge at end of each week)
dev    ← daily working branch (both push here)
```

```bash
# daily workflow
git pull origin dev
# ... write code ...
git add .
git commit -m "describe what you did"
git push origin dev
```

---

## License

MIT