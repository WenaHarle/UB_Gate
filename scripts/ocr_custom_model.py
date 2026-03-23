import pytesseract
from PIL import Image
import os

tesseract_cmd = os.getenv("TESSERACT_CMD")
if tesseract_cmd:
	pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

# Load your image
image = Image.open('tes.jpg')

# Specify the custom trained model for Tesseract
custom_config = r'--oem 1 --psm 6 -l myfont'

# Run Tesseract OCR
text = pytesseract.image_to_string(image, config=custom_config)

print(text)
