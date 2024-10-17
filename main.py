# main.py
import cv2
import time
import roboflow_model
import optical_flow_model
import numpy as np

# Load Roboflow model
model = roboflow_model.load_roboflow_model()

# Initialize video capture
video_capture = cv2.VideoCapture("rocket_launch.mp4")

# Capture the first frame
ret, old_frame = video_capture.read()
if not ret:
    print("Error reading video")
    exit()

# Get the rocket bounding box from the Roboflow model
bbox = roboflow_model.get_rocket_bbox(old_frame, model)
if bbox is None:
    print("No rocket detected in the first frame")
    exit()

x, y, width, height = bbox
rocket_region = old_frame[int(y - height / 2):int(y + height / 2), int(x - width / 2):int(x + width / 2)]
rocket_region = cv2.resize(rocket_region, (100, 100))  # Resize for consistency

# Initialize optical flow
old_gray, p0 = optical_flow_model.initialize_optical_flow(rocket_region)
if p0 is None:
    print("No points to track in the rocket region")
    exit()

# Create a mask for drawing
mask = np.zeros_like(old_frame)

# Main loop to track the rocket
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Get the rocket bounding box from Roboflow model for the current frame
    bbox = roboflow_model.get_rocket_bbox(frame, model)
    if bbox is None:
        print("No rocket detected in this frame, skipping...")
        continue

    x, y, width, height = bbox
    rocket_region = frame[int(y - height / 2):int(y + height / 2), int(x - width / 2):int(x + width / 2)]
    rocket_region = cv2.resize(rocket_region, (100, 100))

    # Track the rocket using optical flow
    frame_gray, good_new, good_old = optical_flow_model.track_optical_flow(old_gray, rocket_region, p0)
    if good_new is None or good_old is None:
        print("Optical flow tracking failed, skipping this frame...")
        continue

    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a + x - width / 2), int(b + y - height / 2)),
                             (int(c + x - width / 2), int(d + y - height / 2)), color=(0, 255, 0), thickness=2)
        frame = cv2.circle(frame, (int(a + x - width / 2), int(b + y - height / 2)), 5, color=(0, 0, 255), thickness=-1)

    # Overlay the mask on the frame
    output = cv2.add(frame, mask)

    # Save the output frame as an image
    cv2.imwrite('output_frame.jpg', output)

    # Update the previous frame and points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    # Add a short delay instead of cv2.waitKey()
    time.sleep(0.03)  # 30ms delay

# Release video capture
video_capture.release()