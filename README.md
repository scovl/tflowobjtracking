# tflowobjtracking

## Object Detection and Tracking with TensorFlow and OpenCV

This project demonstrates how to use TensorFlow and OpenCV for object detection and tracking in real-time. The script `detector.py` captures the screen, detects objects (specifically people), and tracks the detected objects using a tracking algorithm.

## Requirements

- Python 3.x
- TensorFlow
- OpenCV
- NumPy
- mss

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/detector.git
   cd detector
   ```

2. Install the required packages:

   ```bash
   pip install tensorflow opencv-python-headless numpy mss
   ```

## Usage

1. Make sure you have the required libraries installed.
2. Run the `detector.py` script:

   ```bash
   python detector.py
   ```

3. The script will capture the screen, detect objects (people), and track the detected objects.

## Script Explanation

### detector.py

The script performs the following tasks:

1. **Load the Pre-trained Model**:
   - The script loads the SSD MobileNet V2 model from TensorFlow Hub for object detection.

2. **Set Up Screen Capture**:
   - The script uses the `mss` library to capture a specified region of the screen.

3. **Define the Field of View (FOV)**:
   - The FOV region is defined to focus on the central part of the screen.

4. **Set the Desired FPS**:
   - The script aims to run at 60 frames per second (FPS).

5. **Capture and Process the Screen**:
   - The screen is captured, and the image is resized for processing.

6. **Object Detection**:
   - The TensorFlow model detects objects (people) in the captured screen region.

7. **Object Tracking**:
   - OpenCV's KCF tracker is used to track the detected objects.

8. **Display the Results**:
   - The detected and tracked objects are displayed on the screen with a pink circle around them.


## Notes

- This script is designed to work on a Windows machine.
- Make sure your environment supports the required CPU instructions for optimal performance with TensorFlow.
- Adjust the `FovX`, `FovY`, and `size_scale` values according to your needs for better results.


