# **Collision Detection using YOLO and DeepSORT**

This repository contains a Python script that demonstrates a collision detection system using YOLOv5 (You Only Look Once) for object detection and DeepSORT for object tracking. The system is designed to detect collisions between cars in video footage.

## **Directly see the results**
1. Add the copy of this colab to your drive - [Colab link](https://colab.research.google.com/drive/1yP7NKlEPJmsNj19Zfrd3xcrbKYT8Rs-0?usp=sharing)
2. Add the shortcut of this shared folder to your drive [Folder shared with you](https://drive.google.com/drive/folders/1S6PHLtKubaEmb4elW0O0IDX8bsusIp-u?usp=sharing)
3. Put the setting to T4 and do run all


## **Usage**

1. Clone the repository and navigate to the project folder.
2. Install the required dependencies using pip: `pip install -r requirements.txt`
3. Download the DeepSORT model weights using the following command: `!git clone https://github.com/ZQPei/deep_sort_pytorch.git && cd deep_sort_pytorch && !pip install -r requirements.txt`
4. Run the `main.py` script using Python: `python main.py`

## **Configuration**

The script uses several parameters that can be adjusted to fine-tune the collision detection system. These parameters include:

* `collision_threshold`: The distance threshold below which two vehicles are considered to be colliding.
* `iou_threshold`: The Intersection over Union (IoU) threshold above which two bounding boxes are considered to be overlapping.
* `size_ratio_threshold`: A bounding box's minimum size ratio to the frame's largest bounding box.

## **Output**

The script generates several output files, including:

* `detections.json`: A JSON file containing the detection results for each frame.
* `collisions.json`: A JSON file containing the collision detection results.
* `output_vid.mp4`: A video file with annotated bounding boxes and collision detection results.
* `frames_of_collision`: A folder containing frames where collisions were detected.

## **Limitations**

The current implementation has some limitations, including:

* Currently, it is based on a defined method; we could also train the model to detect crash if large amount of data is there 
* The system may detect false positives or false negatives depending on the video quality, camera angle, and lighting conditions.
* The system may not generalize well to different scenarios or environments.

## **Future Work**

Several improvements can be made to the system, including:

* Could use any advanced object detection model to detect cars (YOLOv8)
* Could use any advanced tracking algorithm to track cars (BotSORT, DeepocsSORT)
* Could use object detectors that give quadrilateral output rather than rectangles parallel to x and y axis (Florence)

