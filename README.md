# UB_Gate

License plate detection and OCR experiments using Python (OpenCV + Tesseract), with an additional C++ prototype under `ocr_project`.

## Features

- RTSP stream viewer and motion-based frame capture.
- License plate candidate detection from images.
- Plate color classification (black/white/red/yellow).
- OCR extraction using Tesseract.
- Custom Tesseract trained-data usage (`myfont`).
- C++ OCR example with OpenCV + Tesseract + Leptonica.

## Project Structure

```text
UB_Gate/
├── scripts/                     # Python scripts
│   ├── capture_motion_rtsp.py
│   ├── rtsp_stream.py
│   ├── detect_plate_basic.py
│   ├── detect_plate_color.py
│   ├── detect_plate_ocr.py
│   ├── detect_plate_ocr_linux.py
│   ├── ocr_custom_model.py
│   ├── skew_demo.py
│   └── tesseract_check.py
├── docs/
│   ├── PROJECT_STRUCTURE.md
│   ├── TESSERACT_TRAINING.md
│   └── tesseract_training_notes.txt
├── config/
│   └── .env.example
├── images/                      # Sample images
├── captured_cars/               # Captured motion frames (generated)
├── Train Tesseract/             # Custom Tesseract training artifacts
├── ocr_project/                 # C++ project (OpenCV + Tesseract)
├── requirements.txt
└── .gitignore
```

## Requirements

### Python

- Python 3.9+
- Tesseract OCR installed on your system

Install Python packages:

```bash
pip install -r requirements.txt
```

### System Dependencies

- Windows: install Tesseract from UB Mannheim / official installer.
- Ubuntu/Debian:

```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-eng
```

## Quick Start

### 1. Configure environment variables

PowerShell:

```powershell
$env:RTSP_URL = "rtsp://username:password@host:554/Streaming/Channels/101"
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

Bash:

```bash
export RTSP_URL="rtsp://username:password@host:554/Streaming/Channels/101"
export TESSERACT_CMD="/usr/bin/tesseract"
```

### 2. Run scripts

Examples:

```bash
python scripts/rtsp_stream.py
python scripts/capture_motion_rtsp.py
python scripts/detect_plate_basic.py
python scripts/detect_plate_color.py
python scripts/detect_plate_ocr.py
python scripts/ocr_custom_model.py
```

## C++ OCR Project

C++ source is in `ocr_project/main.cpp` with CMake in `ocr_project/CMakeLists.txt`.

Build steps (Linux):

```bash
cd ocr_project
mkdir -p build
cd build
cmake ..
make
./ocr_project
```

## Notes

- Some scripts use hardcoded sample image names (for example `download.jpeg`, `tes.jpg`, `images/Cars358.png`).
- For custom OCR model, ensure `myfont.traineddata` is available in Tesseract `tessdata` path.
- For detailed training steps, see `docs/TESSERACT_TRAINING.md`.
