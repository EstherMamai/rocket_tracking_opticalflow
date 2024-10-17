import cv2
import numpy as np
import preprocessing

# Load the video
video_capture = cv2.VideoCapture('rocket_launch.mp4')  

# Parameters for Shi-Tomasi corner detection (used for finding keypoints)
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Read the first frame of the video
ret, old_frame = video_capture.read()

# Convert the frame to grayscale
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Detect keypoints (features) in the first frame
keypoints = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing the optical flow (lines to indicate movement)
mask = np.zeros_like(old_frame)

while True:
    # Capture the next frame
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert the current frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using Lucas-Kanade method
    new_keypoints, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, keypoints, None, **lk_params)

    # Select good points for tracking
    good_new = new_keypoints[status == 1]
    good_old = keypoints[status == 1]

    # Draw the tracks of the keypoints
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        # Extract the coordinates of the keypoints
        a, b = new.ravel()
        c, d = old.ravel()

       # Draw a line showing the movement of the keypoints
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color=(0, 255, 0), thickness=2)

        # Draw circles on the keypoints
        frame = cv2.circle(frame, (int(a), int(b)), 5, color=(0, 0, 255), thickness=-1)


    # Overlay the tracks on the original frame
    output = cv2.add(frame, mask)

    # Display the result
    cv2.imshow('Rocket Tracking', output)

    # Update the previous frame and keypoints for the next iteration
    old_gray = gray_frame.copy()
    keypoints = good_new.reshape(-1, 1, 2)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the video capture object and close display window
video_capture.release()
cv2.destroyAllWindows()
