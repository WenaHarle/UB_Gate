import cv2
import numpy as np

def preprocess_image(image):
    """Convert image to grayscale and apply some preprocessing steps."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def detect_license_plate(image):
    """Detect the license plate region in the image."""
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
        if 2 < aspect_ratio < 10 and area > 1000:
            license_plate = image[y:y+h, x:x+w]
            return license_plate
    return None

def detect_background_color(license_plate):
    """Detect the background color of the license plate (black, white, red, or yellow)."""
    hsv_plate = cv2.cvtColor(license_plate, cv2.COLOR_BGR2HSV)

    # Define color ranges for black, white, red, and yellow in HSV
    black_lower = np.array([0, 0, 0])
    black_upper = np.array([180, 255, 50])

    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 20, 255])

    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 50, 50])
    red_upper2 = np.array([180, 255, 255])

    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])

    # Create masks for each color
    mask_black = cv2.inRange(hsv_plate, black_lower, black_upper)
    mask_white = cv2.inRange(hsv_plate, white_lower, white_upper)
    mask_red1 = cv2.inRange(hsv_plate, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(hsv_plate, red_lower2, red_upper2)
    mask_yellow = cv2.inRange(hsv_plate, yellow_lower, yellow_upper)

    # Check which mask has the largest non-zero pixels (dominant color)
    if cv2.countNonZero(mask_black) > 0:
        return "Black"
    elif cv2.countNonZero(mask_white) > 0:
        return "White"
    elif cv2.countNonZero(mask_red1) > 0 or cv2.countNonZero(mask_red2) > 0:
        return "Red"
    elif cv2.countNonZero(mask_yellow) > 0:
        return "Yellow"
    else:
        return "Unknown"

def main():
    # Load image
    image = cv2.imread('images/Cars358.png')
    
    # Detect license plate
    license_plate = detect_license_plate(image)
    
    if license_plate is not None:
        # Detect the background color of the license plate
        background_color = detect_background_color(license_plate)
        print(f"Detected license plate background color: {background_color}")
        
        # Save the detected license plate to a file
        output_path = 'detected_license_plate.png'
        cv2.imwrite(output_path, license_plate)
        print(f"License plate detected and saved to {output_path}")
    else:
        print("License plate not detected")

if __name__ == "__main__":
    main()
