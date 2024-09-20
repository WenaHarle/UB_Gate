import cv2
import os
import time

# RTSP URL
rtsp_url = 'rtsp://admin:Super123!@10.39.93.120:554/Streaming/Channels/101'

# Create a directory to save captured images
output_dir = 'captured_cars'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Minimum size (in pixels) for detected objects to be captured
MIN_SIZE_THRESHOLD = 300000  # Adjust this value based on expected car size

# Capture the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Cannot open RTSP stream")
    exit()

# Read the first frame to initialize the background
ret, frame1 = cap.read()
frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
frame1_gray = cv2.GaussianBlur(frame1_gray, (21, 21), 0)

frame_count = 0  # To count and name saved images

while cap.isOpened():
    ret, frame2 = cap.read()

    # If frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert the current frame to grayscale and blur it
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.GaussianBlur(frame2_gray, (21, 21), 0)

    # Compute the absolute difference between the current frame and the first frame
    diff_frame = cv2.absdiff(frame1_gray, frame2_gray)

    # Apply a threshold to get a binary image (motion regions)
    _, thresh_frame = cv2.threshold(diff_frame, 25, 255, cv2.THRESH_BINARY)

    # Dilate the thresholded image to fill in holes, making the object appear solid
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # Find contours (i.e., regions where motion is detected)
    contours, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    car_detected = False

    # Draw rectangles around the contours if they are large enough
    for contour in contours:
        if cv2.contourArea(contour) < MIN_SIZE_THRESHOLD:  # Use minimum size threshold
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        car_detected = True

    # If a car (motion) is detected, save the frame as an image
    if car_detected:
        frame_count += 1
        timestamp = time.strftime("%Y%m%d-%H%M%S")  # Generate a timestamp
        image_path = os.path.join(output_dir, f"car_{frame_count}_{timestamp}.jpg")
        cv2.imwrite(image_path, frame2)  # Save the frame as a JPG image
        print(f"Captured and saved image: {image_path}")

    # Display the resulting frame with detected motion
    cv2.imshow('Motion Detection', frame2)

    # Update the previous frame to the current
