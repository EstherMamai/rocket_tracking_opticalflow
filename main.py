# Import required libraries
import cv2
import numpy as np
import time
from roboflow import Roboflow

# Load the Roboflow model
rf = Roboflow(api_key="amLDwnyGfdSsvyilDr0g")
project = rf.workspace().project("rocket-detect")
model = project.version("2").model

# Initialize video capture
video_capture = cv2.VideoCapture("rocket_launch.mp4")

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Capture the first frame
ret, old_frame = video_capture.read()
if not ret:
    print("Error reading video")
    exit()

# Detect the rocket in the first frame using the Roboflow model
cv2.imwrite("frame.jpg", old_frame)
result = model.predict("frame.jpg").json()

# Get the first bounding box coordinates of the detected rocket
rocket_bbox = result['predictions'][0]  # Assuming the first detection is the rocket
x, y, width, height = rocket_bbox['x'], rocket_bbox['y'], rocket_bbox['width'], rocket_bbox['height']

# Crop the detected rocket region from the first frame
rocket_region = old_frame[int(y - height / 2):int(y + height / 2), int(x - width / 2):int(x + width / 2)]
rocket_region = cv2.resize(rocket_region, (100, 100))  # Fixed size ROI for consistency
old_gray = cv2.cvtColor(rocket_region, cv2.COLOR_BGR2GRAY)
# Initialize video capture and get the first frame
ret, frame = video_capture.read()

# Convert the first frame to grayscale
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Create a mask for drawing (same size as frame but black)
mask = np.zeros_like(frame)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Detect good features to track within the rocket region
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Ensure p0 is valid
if p0 is None or len(p0) == 0:
    print("No points to track in the first frame")
else:
    print("Tracking points initialized")

# Main loop to track the rocket
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Update the rocket region in the current frame using the previously tracked points
    rocket_region = frame[int(y - height / 2):int(y + height / 2), int(x - width / 2):int(x + width / 2)]
    rocket_region = cv2.resize(rocket_region, (100, 100))

    # Convert the new rocket region to grayscale
    frame_gray = cv2.cvtColor(rocket_region, cv2.COLOR_BGR2GRAY)
    
    # Ensure image sizes are consistent
    print("Old gray shape:", old_gray.shape)
    print("Frame gray shape:", frame_gray.shape)

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Check if optical flow calculation was successful
    if p1 is None or st is None:
        print("Optical flow calculation failed, skipping this frame")
        continue

    # Select good points for tracking
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Draw the tracks (as in your original code)
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a + x - width / 2), int(b + y - height / 2)), (int(c + x - width / 2), int(d + y - height / 2)), color=(0, 255, 0), thickness=2)
        frame = cv2.circle(frame, (int(a + x - width / 2), int(b + y - height / 2)), 5, color=(0, 0, 255), thickness=-1)

    # Overlay the mask on the frame
    output = cv2.add(frame, mask)

    # Save the output frame as an image
    cv2.imwrite('output_frame.jpg', output)

    # Update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    # Add a short delay instead of cv2.waitKey() since we're not using windows
    time.sleep(0.03)  # 30ms delay

# Release video capture
video_capture.release()