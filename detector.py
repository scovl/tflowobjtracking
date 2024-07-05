import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import time
from mss import mss

# Load the pre-trained model from TensorFlow Hub
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

# Set up screen capture with mss
sct = mss()

# Define the Field of View (FOV) region in pixels
FovX = 800
FovY = 600

# Set the desired FPS
desired_fps = 60
frame_interval = 1.0 / desired_fps  # Interval between frames

# Set the resize factor
size_scale = 2  # Increase the value to reduce the resolution even more

# Get the screen dimensions
screen_width = sct.monitors[1]['width']
screen_height = sct.monitors[1]['height']

# Calculate the central region based on the desired FOV
region = {
    "top": (screen_height // 2) - (FovY // 2),
    "left": (screen_width // 2) - (FovX // 2),
    "width": FovX,
    "height": FovY
}

# Tracking parameters
tracker = cv2.TrackerKCF_create()
tracking = False
bbox = None

while True:
    start_time = time.time()

    # Capture the screen
    sct_img = sct.grab(region)
    ori_img = np.array(sct_img)

    # Convert from RGBA to RGB
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGBA2RGB)

    # Resize the captured image
    ori_img_resized = cv2.resize(ori_img, (ori_img.shape[1] // size_scale, ori_img.shape[0] // size_scale))
    img_h, img_w, _ = ori_img_resized.shape

    if not tracking:
        # Object detection
        image = np.expand_dims(ori_img_resized, 0)
        result = detector(image)
        result = {key: value.numpy() for key, value in result.items()}
        boxes = result['detection_boxes'][0]
        scores = result['detection_scores'][0]
        classes = result['detection_classes'][0]

        # Check each detected object
        detected_boxes = []
        for i, box in enumerate(boxes):
            # Select only people (class: 1) with high confidence
            if classes[i] == 1 and scores[i] >= 0.6:
                ymin, xmin, ymax, xmax = tuple(box)
                left, right, top, bottom = int(xmin * img_w), int(xmax * img_w), int(ymin * img_h), int(ymax * img_h)
                center_x, center_y = (left + right) // 2, (top + bottom) // 2

                # Check if the object's center is close to the screen center
                if abs(center_x - img_w // 2) < FovX // 4 and abs(center_y - img_h // 2) < FovY // 4:
                    detected_boxes.append((left, top, right - left, bottom - top))
                    bbox = (left, top, right - left, bottom - top)
                    tracker = cv2.TrackerKCF_create()  # Recreate the tracker
                    tracker.init(ori_img_resized, bbox)
                    tracking = True
                    break

    else:
        # Track the detected object
        if bbox is not None:
            success, bbox = tracker.update(ori_img_resized)
            if success:
                left, top, w, h = [int(v) for v in bbox]
                center_x, center_y = left + w // 2, top + h // 2
                radius = max(min(w, h) // 4, 10)
                cv2.circle(ori_img_resized, (center_x, center_y), radius, (255, 0, 255), -1)
            else:
                tracking = False

    # Convert to RGB for correct display
    ori_img_resized = cv2.cvtColor(ori_img_resized, cv2.COLOR_BGR2RGB)
    cv2.imshow("Detected Targets", ori_img_resized)

    # Press `q` to stop the program
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Calculate the time to maintain the desired FPS
    elapsed_time = time.time() - start_time
    sleep_time = max(0, frame_interval - elapsed_time)
    time.sleep(sleep_time)

sct.close()
cv2.destroyAllWindows()
