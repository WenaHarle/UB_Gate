import pytesseract
import os

tesseract_cmd = os.getenv("TESSERACT_CMD")
if tesseract_cmd:
	pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

print("Tesseract command:", pytesseract.pytesseract.tesseract_cmd)


