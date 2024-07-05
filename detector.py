import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import time
from mss import mss
import pyglet
from pyglet import shapes
from pyglet.gl import *
import win32api
import win32gui
import win32con
import math

# Load the pre-trained model from TensorFlow Hub
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

# Set up screen capture with mss
sct = mss()

# Define the Field of View (FOV) region in pixels
FovX = 70
FovY = 180

# Set the desired FPS
desired_fps = 30
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

# Initialize Pyglet window for overlay
config = pyglet.gl.Config(double_buffer=True, alpha_size=8)
window = pyglet.window.Window(FovX, FovY, style=pyglet.window.Window.WINDOW_STYLE_BORDERLESS, config=config)
window.set_location((screen_width // 2) - (FovX // 2), (screen_height // 2) - (FovY // 2))
window.set_caption('Detected Targets')
window.set_mouse_visible(False)  # Hide the mouse cursor
batch = pyglet.graphics.Batch()
rectangles = []

# Make the window transparent on Windows
hwnd = window._hwnd
extended_style_settings = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, extended_style_settings | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT)
win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(0, 0, 0), 0, win32con.LWA_COLORKEY)

@window.event
def on_draw():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    window.clear()
    batch.draw()

# Tracking parameters
tracker = None
tracking = False
bbox = None

def update(dt):
    global tracking, bbox, rectangles, tracker

    # Capture the screen
    try:
        sct_img = sct.grab(region)
        ori_img = np.array(sct_img)

        # Convert from BGRA to RGB
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGRA2RGB)

        # Resize the captured image
        ori_img_resized = cv2.resize(ori_img, (ori_img.shape[1] // size_scale, ori_img.shape[0] // size_scale))
        img_h, img_w, _ = ori_img_resized.shape

        rectangles = []

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
            if classes[i] == 1 and scores[i] >= 0.5:
                ymin, xmin, ymax, xmax = tuple(box)
                left, right, top, bottom = int(xmin * img_w), int(xmax * img_w), int(ymin * img_h), int(ymax * img_h)
                detected_boxes.append((left, right, top, bottom))

        #print("Detected:", len(detected_boxes))

        # Check Closest
        if len(detected_boxes) >= 1:
            min_distance = 99999
            closest_index = 0
            centers = []
            for i, box in enumerate(detected_boxes):
                x1, x2, y1, y2 = box
                c_x = ((x2 - x1) / 2) + x1
                c_y = ((y2 - y1) / 2) + y1
                centers.append((c_x, c_y))
                dist = math.sqrt(math.pow(img_w / 2 - c_x, 2) + math.pow(img_h / 2 - c_y, 2))
                if dist < min_distance:
                    min_distance = dist
                    closest_index = i

            # Draw the closest box
            left, right, top, bottom = detected_boxes[closest_index]
            line_thickness = 2
            rect_top = shapes.Line(left, FovY - top, left + (right - left), FovY - top, color=(255, 0, 255), width=line_thickness, batch=batch)
            rect_bottom = shapes.Line(left, FovY - bottom, left + (right - left), FovY - bottom, color=(255, 0, 255), width=line_thickness, batch=batch)
            rect_left = shapes.Line(left, FovY - top, left, FovY - bottom, color=(255, 0, 255), width=line_thickness, batch=batch)
            rect_right = shapes.Line(right, FovY - top, right, FovY - bottom, color=(255, 0, 255), width=line_thickness, batch=batch)
            rectangles.extend([rect_top, rect_bottom, rect_left, rect_right])

    except Exception as e:
        print(f"Error capturing screen: {e}")

# Schedule the update function
pyglet.clock.schedule_interval(update, frame_interval)

# Show the window and run the application
window.set_visible(True)
pyglet.app.run()
