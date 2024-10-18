# Rocket Tracking and Antenna Orientation Control

This project aims to build a system that tracks a rocket's flight path and controls an antenna's orientation to maintain a stable communication link. The tracking system uses optical flow, computer vision techniques, and a pre-trained YOLOv5 model to detect and track the rocket's position in real-time. The system then adjusts the antenna's direction based on the rocket's movements, ensuring consistent data transmission.

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The project consists of a camera system that captures the rocket's flight. Optical flow techniques track the movement of the rocket, while the YOLOv5 model identifies the rocket in each frame. The detected position is then used to orient the antenna to follow the rocket's trajectory and maintain a stable connection with a base station.

Key components include:
- Optical flow for motion tracking
- YOLOv5 for object detection
- Antenna control algorithm based on rocket position
- Real-time video processing

## Technologies Used
- Python 3.12
- OpenCV 4.10.0
- Roboflow (for pre-trained YOLOv5 model)
- Tkinter (for GUI)
- Threading (for real-time tracking)
- Numpy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rocket-tracking-antenna-control.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the Roboflow model:
   - Ensure you have a Roboflow account and access to the trained YOLOv5 model.
   - Update the API key in `roboflow_model.py`.

4. Connect your camera feed, either by using a video file or a live feed.

## Usage

1. Run the GUI for tracking and control:
   ```bash
   python rocket_gui.py
   ```

2. The application will:
   - Load the YOLOv5 model.
   - Start tracking the rocket's position in real-time.
   - Control the antenna based on the rocket's trajectory.

3. Adjust camera input parameters and antenna specifications as needed in the configuration file.

## Model Training

If you'd like to train your own model for tracking, you can follow these steps:
1. Collect video data of rockets in flight.
2. Use Roboflow to annotate the data and train a YOLOv5 model.
3. Integrate your trained model by updating the model paths in the `roboflow_model.py` file.

## Contributing

If you'd like to contribute to this project, feel free to fork the repository and submit a pull request. Please ensure all changes are documented and tested.
