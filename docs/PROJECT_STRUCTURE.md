# Project Structure Guide

This document explains where each part of the repository belongs.

## Directories

- `scripts/`: runnable Python scripts for detection, OCR, and streaming.
- `docs/`: project documentation and training notes.
- `config/`: environment and configuration templates.
- `images/`: static sample images used for testing.
- `captured_cars/`: generated output images from motion detection.
- `Train Tesseract/`: custom Tesseract model training outputs.
- `ocr_project/`: C++ OCR prototype project using CMake.

## Script Map

- `scripts/rtsp_stream.py`: open and display an RTSP stream.
- `scripts/capture_motion_rtsp.py`: motion detection on RTSP and frame capture.
- `scripts/detect_plate_basic.py`: detect and crop a potential plate region.
- `scripts/detect_plate_color.py`: detect plate and classify dominant background color.
- `scripts/detect_plate_ocr.py`: detect plate then OCR using Tesseract (Windows-friendly path style).
- `scripts/detect_plate_ocr_linux.py`: OCR flow variant with matplotlib display.
- `scripts/ocr_custom_model.py`: OCR using custom Tesseract language model (`myfont`).
- `scripts/skew_demo.py`: simple image skew transformation demo.
- `scripts/tesseract_check.py`: minimal Tesseract path setup snippet.

## Suggested Future Cleanup

- Convert script-style files into reusable modules under `src/`.
- Add `tests/` with sample-image-based unit tests.
- Replace hardcoded sample paths with CLI arguments.
- Keep generated images only in `captured_cars/` and avoid committing them.
