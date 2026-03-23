import cv2
import os

rtsp_url = os.getenv("RTSP_URL", "rtsp://<username>:<password>@<host>:554/Streaming/Channels/101")

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Cannot open RTSP stream. Set RTSP_URL environment variable first.")
    exit()

# Loop to continuously read and display frames from the RTSP stream
while cap.isOpened():
    ret, frame = cap.read()

    # If frame is read correctly, ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display the current frame
    cv2.imshow('RTSP Stream', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the display window
cap.release()
cv2.destroyAllWindows()
