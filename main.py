import torch
import cv2
import json
import numpy as np
import utils

def main(video_path, collision_threshold, output_video_path, detections_output, collisions_output, frame_folder):
    # Initialize DeepSORT
    deepsort = init_deepsort()

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    tracks = {}

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # print(frames)


    detection_results = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect vehicles using YOLO
        detections = detect_vehicles(frame)
        outputs = track_vehicles(deepsort, detections, frame)

        detection_dicts = []
        for index, row in detections.iterrows():
            detection_dict = {
                'xmin': row['xmin'],
                'ymin': row['ymin'],
                'xmax': row['xmax'],
                'ymax': row['ymax'],
                'confidence': row['confidence'],
                'class': row['class']
            }
            detection_dicts.append(detection_dict)

        detection_results.append({
            'frame_id': frame_count,
            'detections': detection_dicts
        })

        # Track vehicles and update tracks dictionary
        frame_tracks = []
        for output in outputs:
            track_id = output[-1]
            bbox = output[:4]
            frame_tracks.append((bbox, track_id))
            if track_id not in tracks:
                tracks[track_id] = []
            tracks[track_id].append((*bbox, frame_count))

        # Draw tracked vehicles on the frame
        draw_tracks(frame, frame_tracks)

        # Detect collisions and visualize on frame
        if frame_count%30 ==0 and frame_count>frames//2: #Checking after every 30 frames when greater than half of frames
          collisions = detect_collisions(tracks, base_collision_threshold=collision_threshold, iou_threshold=0.05, size_ratio_threshold=0.1)

          # Saving the frame of collision
          for collision in collisions:
              # Highlight collision location on frame
              collision_location = collision['location']
              cv2.circle(frame, collision_location, 10, (0, 0, 255), -1)  # Red circle at collision point

              # Save the frame where collision occurs (optional)
              name = frame_folder + '/' + str(frame_count)+'.jpg'
              cv2.imwrite(name, frame)

        # Write frame with annotations to output video
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    # Save detection results to JSON file
    with open(detections_output, 'w') as f:
        json.dump(detection_results, f)

    collisions = detect_collisions(tracks, base_collision_threshold=500, iou_threshold=0.05, size_ratio_threshold=0.1)


    # Format collision results and save to JSON file
    collision_results = format_results(collisions)
    for entry in collision_results:
      entry['id1'] = int(entry['id1'])
      entry['id2'] = int(entry['id2'])
      entry['location'] = tuple(map(int, entry['location']))
    with open(collisions_output, 'w') as f:
        json.dump(collision_results, f)

    print("Processing complete. Outputs saved.")
    return tracks
