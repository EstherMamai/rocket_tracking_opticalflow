# roboflow_model.py
import cv2
from roboflow import Roboflow

def load_roboflow_model():
    rf = Roboflow(api_key="amLDwnyGfdSsvyilDr0g")
    project = rf.workspace().project("rocket-detect")
    model = project.version("2").model
    return model

def get_rocket_bbox(frame, model):
    # Save the frame as an image and make predictions
    cv2.imwrite("temp_frame.jpg", frame)
    result = model.predict("temp_frame.jpg").json()

    # Get the first bounding box coordinates of the detected rocket
    if result['predictions']:
        rocket_bbox = result['predictions'][0]  # Assuming the first detection is the rocket
        return rocket_bbox['x'], rocket_bbox['y'], rocket_bbox['width'], rocket_bbox['height']
    else:
        return None
