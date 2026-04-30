import math
import os
import cv2
from ultralytics import YOLO

def extract_dataset(video_path, output_dir, completed_queue=None):
    """Extracts photos of moving cars. When a car track goes stale (leaves the frame),
    it pushes the completed folder path to the provided queue for immediate labeling."""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = YOLO("yolov8n.pt")
    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        print(f"ERROR: Video not loaded: {video_path}")
        return

    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    
    track_history = {}
    frames_saved = {}
    moving_cars_stats = set()
    unconfirmed_crops = {}

    history_length = 10
    movement_threshold = 50
    
    last_seen_frame = {}
    completed_tracks = set()
    frame_count = 0
    TIMEOUT_FRAMES = 30

    def save_car_image(crop_img, t_id, frame_num):
        if crop_img.size > 0:
            car_dir = os.path.join(output_dir, f"{video_basename}_{t_id:04d}")
            if not os.path.exists(car_dir):
                os.makedirs(car_dir)
            
            filename = os.path.join(car_dir, f"{frame_num:04d}.jpg")
            cv2.imwrite(filename, crop_img)

    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            break
        
        frame_count += 1
        results = model.track(frame, classes=[2], persist=True, verbose=False)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                last_seen_frame[track_id] = frame_count

                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                car_crop = frame[y1:y2, x1:x2]

                if track_id not in track_history:
                    track_history[track_id] = []

                track_history[track_id].append((cx, cy))
                if len(track_history[track_id]) > history_length:
                    track_history[track_id].pop(0)

                moving = False

                if len(track_history[track_id]) == history_length:
                    old_x, old_y = track_history[track_id][0]
                    distance = math.hypot(cx - old_x, cy - old_y)

                    if distance > movement_threshold:
                        moving = True
                        moving_cars_stats.add(track_id)

                if track_id not in frames_saved:
                    frames_saved[track_id] = 0

                if moving:
                    if track_id in unconfirmed_crops and len(unconfirmed_crops[track_id]) > 0:
                        for buffered_crop in unconfirmed_crops[track_id]:
                            save_car_image(buffered_crop, track_id, frames_saved[track_id])
                            frames_saved[track_id] += 1
                        unconfirmed_crops[track_id].clear()

                    save_car_image(car_crop, track_id, frames_saved[track_id])
                    frames_saved[track_id] += 1
                else:
                    if track_id not in unconfirmed_crops:
                        unconfirmed_crops[track_id] = []
                    unconfirmed_crops[track_id].append(car_crop)

                    if len(unconfirmed_crops[track_id]) > history_length:
                        unconfirmed_crops[track_id].pop(0)

        for t_id, last_seen in list(last_seen_frame.items()):
            if t_id not in completed_tracks and (frame_count - last_seen > TIMEOUT_FRAMES):
                completed_tracks.add(t_id)
                if completed_queue is not None:
                    car_dir = os.path.join(output_dir, f"{video_basename}_{t_id:04d}")
                    if os.path.exists(car_dir):
                        completed_queue.put(car_dir)

    capture.release()
    
    for t_id in list(last_seen_frame.keys()):
        if t_id not in completed_tracks:
            completed_tracks.add(t_id)
            if completed_queue is not None:
                car_dir = os.path.join(output_dir, f"{video_basename}_{t_id:04d}")
                if os.path.exists(car_dir):
                    completed_queue.put(car_dir)