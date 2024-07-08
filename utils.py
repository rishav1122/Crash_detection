import torch
import cv2
import numpy as np
import pandas as pd
import os
import gdown
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')
# Define the path to the DeepSORT weights
REID_CKPT = "deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7"

# # Function to download DeepSORT weights if not present
# def download_deepsort_weights():
#     if not os.path.exists(REID_CKPT):
#         os.makedirs(os.path.dirname(REID_CKPT), exist_ok=True)
#         url = "https://drive.google.com/uc?id=1_6PxmAcjS6iR2Y4StbVZG6gLPZ1U7v-W"
#         gdown.download(url, REID_CKPT, quiet=False)
#         print("DeepSORT weights downloaded.")

# Initialize YOLOv5 model

# Initialize DeepSORT
def init_deepsort():
    cfg = get_config()
    cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
    deepsort = DeepSort(
        REID_CKPT,
        max_dist=cfg.DEEPSORT.MAX_DIST,
        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg.DEEPSORT.MAX_AGE,
        n_init=cfg.DEEPSORT.N_INIT,
        nn_budget=cfg.DEEPSORT.NN_BUDGET,
        use_cuda=True
    )
    return deepsort

def detect_vehicles(frame):
    results = model(frame)
    detections = results.pandas().xyxy[0]  # Convert results to pandas DataFrame

    # Filter detections for the "car" class
    car_detections = detections[detections['name'] == 'car']

    return car_detections

def track_vehicles(deepsort, detections, frame):
    bbox_xywh = []
    confs = []
    classes = []  # List to store object classes

    for index, row in detections.iterrows():

            x1, y1, x2, y2 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            bbox_xywh.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
            confs.append(row['confidence'])
            classes.append(row['class'])
            # print(x1, y1, x2, y2)

    if len(bbox_xywh) == 0:
        return np.empty((0, 6), dtype=np.int32)  # Return an empty array if no cars detected

    bbox_xywh = np.array(bbox_xywh)
    confs = np.array(confs)
    classes = np.array(classes)

    # Ensure frame is converted to BGR format (as required by cv2)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Call DeepSort update method
    outputs, _ = deepsort.update(bbox_xywh, confs, classes, frame_bgr)

    return outputs


def calculate_iou(box1, box2):
    # Calculate the intersection area
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # Calculate the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the union area
    union_area = box1_area + box2_area - inter_area

    # Calculate the IoU
    iou = inter_area / union_area if union_area > 0 else 0

    return iou

def detect_collisions(tracks, base_collision_threshold, iou_threshold=0.2, size_ratio_threshold=0.1):
    collisions = []

    # Extracting track IDs and their positions
    track_ids = list(tracks.keys())

    # Find the size of the largest vehicle in the frame
    max_size = 0
    for track_id in track_ids:
        last_pos = tracks[track_id][-1]
        size = (last_pos[2] - last_pos[0]) * (last_pos[3] - last_pos[1])
        if size > max_size:
            max_size = size

    # Iterate through each pair of different tracks
    for i in range(len(track_ids)):
        for j in range(i + 1, len(track_ids)):
            track_id1 = track_ids[i]
            track_id2 = track_ids[j]

            # Get the last positions of the two tracks
            last_pos1 = tracks[track_id1][-1]
            last_pos2 = tracks[track_id2][-1]

            # Calculate centers of the bounding boxes
            center1 = np.array([(last_pos1[0] + last_pos1[2]) / 2, (last_pos1[1] + last_pos1[3]) / 2])
            center2 = np.array([(last_pos2[0] + last_pos2[2]) / 2, (last_pos2[1] + last_pos2[3]) / 2])
            distance = np.linalg.norm(center1 - center2)

            # Calculate the sizes (areas) of the bounding boxes
            size1 = (last_pos1[2] - last_pos1[0]) * (last_pos1[3] - last_pos1[1])
            size2 = (last_pos2[2] - last_pos2[0]) * (last_pos2[3] - last_pos2[1])

            # Adjust the collision threshold based on the sizes of the bounding boxes
            adjusted_threshold = base_collision_threshold * ((size1 + size2) ** 0.5) / 500  # Adjusting the scaling factor

            # Calculate IoU
            iou = calculate_iou(last_pos1, last_pos2)

            # Check size ratio relative to the largest vehicle
            size_ratio1 = size1 / max_size
            size_ratio2 = size2 / max_size

            # Check if distance is less than the adjusted collision threshold, IoU is greater than the threshold,
            # and both vehicles are reasonably large compared to the largest vehicle in the frame
            if distance < adjusted_threshold and iou > iou_threshold and size_ratio1 > size_ratio_threshold and size_ratio2 > size_ratio_threshold:
                collision_point = ((last_pos1[0] + last_pos2[0]) // 2, (last_pos1[1] + last_pos2[1]) // 2)
                collisions.append({'id1': track_id1, 'id2': track_id2, 'location': collision_point})
                # print(track_id1, track_id2, distance, adjusted_threshold, iou, size_ratio1, size_ratio2)

    return collisions



def format_results(collisions):
    return collisions

def draw_tracks(frame, tracks):
    # print("frame",frame.shape)
    # print("tracks",tracks)
    for bbox, track_id in tracks:

        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, str(track_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
