**Collision Detection using YOLO and DeepSORT**

This repository contains a Python script that demonstrates a collision detection system using YOLO (You Only Look Once) for object detection and DeepSORT for object tracking. The system is designed to detect collisions between vehicles in video footage.

**Requirements**

* Python 3.x
* PyTorch
* OpenCV
* NumPy
* Pandas
* Google Colab (for running the code)

**Usage**

1. Clone the repository and navigate to the project folder.
2. Install the required dependencies using pip: `pip install -r requirements.txt`
3. Download the DeepSORT model weights using the following command: `!git clone https://github.com/ZQPei/deep_sort_pytorch.git && cd deep_sort_pytorch && !pip install -r requirements.txt`
4. Run the `main.py` script using Python: `python main.py`

**Configuration**

The script uses several parameters that can be adjusted to fine-tune the collision detection system. These parameters include:

* `collision_threshold`: The distance threshold below which two vehicles are considered to be colliding.
* `iou_threshold`: The Intersection over Union (IoU) threshold above which two bounding boxes are considered to be overlapping.
* `size_ratio_threshold`: The minimum size ratio of a bounding box to the largest bounding box in the frame.

**Output**

The script generates several output files, including:

* `detections.json`: A JSON file containing the detection results for each frame.
* `collisions.json`: A JSON file containing the collision detection results.
* `output_vid.mp4`: A video file with annotated bounding boxes and collision detection results.
* `frames_of_collision`: A folder containing frames where collisions were detected.

**Limitations**

The current implementation has some limitations, including:

* The system may detect false positives or false negatives depending on the video quality, camera angle, and lighting conditions.
* The system may not generalize well to different scenarios or environments.

**Future Work**

Several improvements can be made to the system, including:

* Improving the object detection and tracking algorithms to reduce false positives and false negatives.
* Integrating additional sensors or data sources, such as GPS or lidar, to improve the accuracy of the system.
* Expanding the system to detect collisions between different types of objects, such as pedestrians or cyclists.

**License**

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for details.
