import pytesseract
from PIL import Image

# Specify the path to the Tesseract executable if needed (for Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load your image
image = Image.open('tes.jpg')

# Specify the custom trained model for Tesseract
custom_config = r'--oem 1 --psm 6 -l myfont'

# Run Tesseract OCR
text = pytesseract.image_to_string(image, config=custom_config)

print(text)
