# Custom Object Detection

This project demonstrates how to train a custom object detection model tailored to identify specific targets in video games. The primary focus is on detecting enemy characters in gameplay footage, leveraging TensorFlow and a SSD Mobilenet model.

## Project Structure

Here's a breakdown of the project directory:

```
project/
├── models/
│   └── my_ssd_mobilenet_v2/
│       ├── pipeline.config
│       ├── train/
│       └── eval/
├── imgs/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── test/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
├── annotations/
│   ├── train_labels.csv
│   ├── test_labels.csv
│   ├── train.record
│   └── test.record
├── scripts/
│   ├── generate_tfrecord.py
│   ├── xml_to_csv.py
│   └── label_map.pbtxt
└── pre-trained-model/
    └── ssd_mobilenet_v2/
        └── checkpoint/
            ├── checkpoint
            ├── ckpt-0.data-00000-of-00001
            └── ckpt-0.index
```

## Prerequisites

- Python 3.7+
- TensorFlow 2.x
- TensorFlow Object Detection API
- Pyglet
- OpenCV
- MSS (for screen capture)

## Setup Instructions

1. **Install Dependencies**:
    - Ensure TensorFlow 2.x is installed along with other Python libraries such as `numpy`, `pandas`, `opencv-python`, and `pyglet`.

2. **Prepare Data**:
    - Collect images from your game footage and annotate them using tools like LabelImg.
    - Save annotated images in the `imgs/train` and `imgs/test` directories.

3. **Convert Annotations**:
    - Run `xml_to_csv.py` to convert XML annotations to CSV format.
    - Run `generate_tfrecord.py` to convert these CSV files into TFRecord format, which is required by the TensorFlow Object Detection API.

   ```bash
   python scripts/xml_to_csv.py
   python scripts/generate_tfrecord.py --csv_input=annotations/train_labels.csv --image_dir=imgs/train --output_path=annotations/train.record
   python scripts/generate_tfrecord.py --csv_input=annotations/test_labels.csv --image_dir=imgs/test --output_path=annotations/test.record
   ```

4. **Modify the Pipeline Configuration**:
    - Download and modify the `pipeline.config` for the SSD Mobilenet model from TensorFlow Model Zoo. Update paths for the `train.record`, `test.record`, `label_map.pbtxt`, and the pre-trained checkpoint.

5. **Training the Model**:
    - Start the model training process by executing the command below. Adjust paths according to your setup.

    ```bash
    python models/research/object_detection/model_main_tf2.py \
        --model_dir=models/my_ssd_mobilenet_v2 \
        --pipeline_config_path=models/my_ssd_mobilenet_v2/pipeline.config
    ```

    This will train your model based on the custom dataset. Monitor training progress via TensorBoard.

6. **Evaluate and Deploy**:
    - After training, evaluate the model's performance. If satisfied, integrate the model into your detection script (`detector.py`) to test its effectiveness in real-time or on recorded gameplay footage.