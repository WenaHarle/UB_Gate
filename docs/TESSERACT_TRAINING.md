# Tesseract Custom Training Notes

This project includes a custom trained model (`myfont`) for OCR experiments.

## Prerequisites (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-eng
sudo apt install imagemagick fontforge
```

## Training Commands

```bash
text2image --fonts_dir=. --font='Indonesia License Plate' --outputbase=myfont.exp0 --text=myfont.txt --ptsize 12

tesseract myfont.exp0.tif myfont.exp0 nobatch box.train

unicharset_extractor myfont.exp0.box

# Create file: font_properties
# Content: myfont 0 0 0 0 0

shapeclustering -F font_properties -U unicharset myfont.exp0.tr
mftraining -F font_properties -U unicharset -O myfont.unicharset myfont.exp0.tr
cntraining myfont.exp0.tr

mv inttemp myfont.inttemp
mv normproto myfont.normproto
mv pffmtable myfont.pffmtable
mv shapetable myfont.shapetable
```

## Output Files

Expected generated files include:

- `myfont.inttemp`
- `myfont.normproto`
- `myfont.pffmtable`
- `myfont.shapetable`
- `myfont.unicharset`
- `myfont.traineddata`

## Usage

Use the model from Python with config similar to:

```python
custom_config = r'--oem 1 --psm 6 -l myfont'
```

Ensure `myfont.traineddata` is available in your Tesseract `tessdata` directory or configured via `TESSDATA_PREFIX`.
