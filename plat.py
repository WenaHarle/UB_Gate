import cv2
import numpy as np

def preprocess_image(image):
    """ Convert image to grayscale and apply some preprocessing steps. """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def detect_license_plate(image):
    """ Detect the license plate region in the image. """
    processed = preprocess_image(image)
    edges = cv2.Canny(processed, 30, 150)
    
    # Find contours in the edged image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Approximate the contour and get bounding box
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)
        
        # Filter out potential license plate regions based on aspect ratio and area
        aspect_ratio = w / float(h)
        area = cv2.contourArea(contour)
        if 2 < aspect_ratio < 6 and area > 1000:
            license_plate = image[y:y+h, x:x+w]
            return license_plate
    return None

def main():
    # Load image
    image = cv2.imread('images/Cars358.png')
    
    # Detect license plate
    license_plate = detect_license_plate(image)
    
    if license_plate is not None:
        # Save the detected license plate to a file
        output_path = 'detected_license_plate.png'
        cv2.imwrite(output_path, license_plate)
        print(f"License plate detected and saved to {output_path}")
    else:
        print("License plate not detected")

if __name__ == "__main__":
    main()