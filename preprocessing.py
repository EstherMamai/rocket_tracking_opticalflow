import cv2
import numpy as np

# Open video file or capture device
video_capture = cv2.VideoCapture('rocket_launch.mp4')

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Read the first frame
ret, old_frame = video_capture.read()

# Step 1: Convert the first frame to grayscale
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Step 2: Apply Gaussian Blur to reduce noise
blurred_frame = cv2.GaussianBlur(old_gray, (5, 5), 0)

# Step 3: Detect keypoints (features) in the first frame
keypoints = cv2.goodFeaturesToTrack(blurred_frame, mask=None, **feature_params)

# Create a mask image for drawing optical flow
mask = np.zeros_like(old_frame)

# Loop through video frames
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Step 4: Convert the current frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 5: Apply Gaussian Blur to reduce noise in current frame
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Step 6: Calculate optical flow
    new_keypoints, status, error = cv2.calcOpticalFlowPyrLK(old_gray, blurred_frame, keypoints, None, **lk_params)

    # Select good points for tracking
    good_new = new_keypoints[status == 1]
    good_old = keypoints[status == 1]

    # Draw the tracks of the keypoints
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()

        # Draw lines and circles for tracking movement
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color=(0, 255, 0), thickness=2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, color=(0, 0, 255), thickness=-1)

    # Overlay the tracks on the original frame
    output = cv2.add(frame, mask)

    # Display the result
    cv2.imshow('Rocket Tracking', output)

    # Update old frame and keypoints
    old_gray = blurred_frame.copy()
    keypoints = good_new.reshape(-1, 1, 2)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release video capture and close windows
video_capture.release()
cv2.destroyAllWindows()