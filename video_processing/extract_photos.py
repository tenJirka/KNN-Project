import math
import os

import cv2
from ultralytics import YOLO

INPUT_VIDEOS_FOLDER = "videa_fit"
OUTPUT_BASE_DIR = "extracted_photos"
PROGRESS_FILE = "extraction_progress.txt"


def extract_dataset(video_path, output_dir):
    """Extracts photos of moving cars from the video and saves them in the output directory"""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = YOLO("yolov8n.pt")  # TODO: maybe change the model, works fine for now
    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        print("ERROR: Video not loaded")
        return

    track_history = {}
    frames_saved = {}
    moving_cars_stats = set()

    unconfirmed_crops = {}

    history_length = 10
    movement_threshold = 50

    def save_car_image(crop_img, t_id, frame_count):
        """Saves the cropped image to the output directory under a folder with track ID"""

        if crop_img.size > 0:
            car_dir = os.path.join(output_dir, f"{t_id:04d}")
            if not os.path.exists(car_dir):
                os.makedirs(car_dir)

            filename = os.path.join(car_dir, f"{frame_count:04d}.jpg")
            cv2.imwrite(filename, crop_img)

    while capture.isOpened():
        # we go through the video fram by frame, detecting cars and tracking
        success, frame = capture.read()
        if not success:
            break

        results = model.track(frame, classes=[2], persist=True, verbose=False)

        if results[0].boxes.id is not None:
            # we get the bounding boxes and track ids for the cars
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # for each car
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                car_crop = frame[y1:y2, x1:x2]

                if track_id not in track_history:
                    track_history[track_id] = []

                track_history[track_id].append((cx, cy))
                if len(track_history[track_id]) > history_length:
                    track_history[track_id].pop(0)

                # we need to determine if car is moving to disqualify parked cars
                moving = False

                # check the distance if track history is long enough
                if len(track_history[track_id]) == history_length:
                    old_x, old_y = track_history[track_id][0]
                    distance = math.hypot(cx - old_x, cy - old_y)

                    if distance > movement_threshold:
                        moving = True
                        moving_cars_stats.add(track_id)

                if track_id not in frames_saved:
                    frames_saved[track_id] = 0

                if moving:
                    # the car started to move buffer of crops is moved to the output
                    if (
                        track_id in unconfirmed_crops
                        and len(unconfirmed_crops[track_id]) > 0
                    ):
                        for buffered_crop in unconfirmed_crops[track_id]:
                            save_car_image(
                                buffered_crop, track_id, frames_saved[track_id]
                            )
                            frames_saved[track_id] += 1

                        unconfirmed_crops[track_id].clear()

                    # current crop is saved
                    save_car_image(car_crop, track_id, frames_saved[track_id])
                    frames_saved[track_id] += 1

                else:
                    # if the car is not moving we buffer in case it starts moving later
                    if track_id not in unconfirmed_crops:
                        unconfirmed_crops[track_id] = []

                    unconfirmed_crops[track_id].append(car_crop)

                    if len(unconfirmed_crops[track_id]) > history_length:
                        unconfirmed_crops[track_id].pop(0)

    capture.release()
    print(f"Byla zachycena data pro {len(moving_cars_stats)} unikátních jedoucích aut.")


if __name__ == "__main__":
    if not os.path.exists(INPUT_VIDEOS_FOLDER):
        print(f"ERROR: Input folder {INPUT_VIDEOS_FOLDER} not found")

    processed_videos = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            processed_videos = set(f.read().splitlines())

    for video_filename in sorted(os.listdir(INPUT_VIDEOS_FOLDER)):
        if video_filename in processed_videos:
            continue

        video_path = os.path.join(INPUT_VIDEOS_FOLDER, video_filename)
        
        if not os.path.isfile(video_path):
            continue

        video_name_without_ext = os.path.splitext(video_filename)[0]
        output_dir = os.path.join(OUTPUT_BASE_DIR, f"photos_from_{video_name_without_ext}")

        print(f"Processing video: {video_filename}")
        extract_dataset(video_path, output_dir)

        with open(PROGRESS_FILE, "a") as f:
            f.write(video_filename + "\n")