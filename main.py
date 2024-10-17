import cv2
import numpy as np
from roboflow import Roboflow

# Initialize Roboflow API
rf = Roboflow(api_key="amLDwnyGfdSsvyilDr0g")  # Replace with your actual API key
project = rf.workspace().project("rocket-detect")  # Replace with your project name
model = project.version(2).model  # Replace '1' with the correct version number of the model

# Open video file or capture device
video_capture = cv2.VideoCapture('rocket_launch.mp4')

# Parameters for Lucas-Kanade Optical Flow
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
old_gray = cv2.cvtColor(rocket_region, cv2.COLOR_BGR2GRAY)

# Detect good features to track within the rocket region (you can use cv2.goodFeaturesToTrack or other methods)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Create a mask image for drawing (for visualization)
mask = np.zeros_like(old_frame)

while True:
    # Read a new frame
    ret, frame = video_capture.read()
    if not ret:
        break

    # Save the frame to a temporary file for Roboflow API
    cv2.imwrite("frame.jpg", frame)
    
    # Perform rocket detection
    result = model.predict("frame.jpg").json()

    # Get the bounding box of the detected rocket
    if len(result['predictions']) > 0:
        rocket_bbox = result['predictions'][0]
        x, y, width, height = rocket_bbox['x'], rocket_bbox['y'], rocket_bbox['width'], rocket_bbox['height']

        # Crop the detected rocket region from the current frame
        rocket_region = frame[int(y - height / 2):int(y + height / 2), int(x - width / 2):int(x + width / 2)]
        frame_gray = cv2.cvtColor(rocket_region, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow between the previous and current frames
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points for tracking
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a + x - width / 2), int(b + y - height / 2)), (int(c + x - width / 2), int(d + y - height / 2)), color=(0, 255, 0), thickness=2)
            frame = cv2.circle(frame, (int(a + x - width / 2), int(b + y - height / 2)), 5, color=(0, 0, 255), thickness=-1)

        # Overlay the mask on the frame
        output = cv2.add(frame, mask)  # <-- Define output here by adding the frame and mask

        # Save the output frame as an image
        cv2.imwrite('output_frame.jpg', output)

        # Update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    # Exit the loop if 'q' is pressed
    #if cv2.waitKey(30) & 0xFF == ord('q'):
        #break

# Release video capture and close windows
video_capture.release()
cv2.destroyAllWindows()