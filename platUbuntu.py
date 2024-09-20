import cv2
import pytesseract
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('download.jpeg')

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Reduce noise
gray = cv2.bilateralFilter(gray, 11, 17, 17)

# Edge detection
edged = cv2.Canny(gray, 30, 200)

# Find contours
contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

# Find contour of the license plate
license_plate = None
for contour in contours:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
    if len(approx) == 4:
        license_plate = approx
        break

# Crop the license plate from the image
if license_plate is not None:
    x, y, w, h = cv2.boundingRect(license_plate)
    plate_img = gray[y:y + h, x:x + w]

    # Display license plate image using matplotlib
    plt.imshow(plate_img, cmap='gray')
    plt.title("Detected License Plate")
    plt.show()

    # OCR the license plate
    text = pytesseract.image_to_string(plate_img, config='--psm 8')
    print("Detected License Plate Text:", text)

else:
    print("License plate not found")

