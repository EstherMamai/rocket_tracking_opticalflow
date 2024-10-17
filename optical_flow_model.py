# optical_flow_model.py
import cv2
import numpy as np

def initialize_optical_flow(rocket_region):
    old_gray = cv2.cvtColor(rocket_region, cv2.COLOR_BGR2GRAY)

    # Detect good features to track
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    return old_gray, p0

def track_optical_flow(old_gray, rocket_region, p0):
    frame_gray = cv2.cvtColor(rocket_region, cv2.COLOR_BGR2GRAY)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is None or st is None:
        return None, None, None

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    return frame_gray, good_new, good_old