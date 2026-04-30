import os
import glob
import threading
import queue
from extract_photos import extract_dataset
from label_dataset import process_vehicle_folder

INPUT_VIDEO_DIR = "videa_fit"
INTERMEDIATE_DIR = "extracted_car_tracks"
FINAL_DATASET_DIR = "labeled_fit_photos"

def label_worker(q):
    """Background thread that pulls finished folders from the queue and labels them."""
    while True:
        folder_path = q.get()
        if folder_path is None:
            q.task_done()
            break
        
        try:
            process_vehicle_folder(folder_path, FINAL_DATASET_DIR)
        except Exception as e:
            print(f"Error during labeling of {folder_path}: {e}")
        finally:
            q.task_done()

def main():
    if not os.path.exists(INPUT_VIDEO_DIR):
        print(f"Input directory '{INPUT_VIDEO_DIR}' does not exist")
        return

    video_extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv")
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(INPUT_VIDEO_DIR, ext)))
        video_files.extend(glob.glob(os.path.join(INPUT_VIDEO_DIR, ext.upper())))

    if not video_files:
        print(f"No videos found in '{INPUT_VIDEO_DIR}'")
        return

    track_queue = queue.Queue()
    
    label_thread = threading.Thread(target=label_worker, args=(track_queue,), daemon=True)
    label_thread.start()

    for video_path in video_files:
        print(f"PROCESSING VIDEO {video_path}")

        extract_dataset(video_path, INTERMEDIATE_DIR, completed_queue=track_queue)
        
        print(f"VIDEO {video_path} PROCESSED")

    track_queue.put(None)
    
    track_queue.join()
    label_thread.join()
    
    print("===============================")
    print("====== PIPELINE COMPLETE ======")
    print("===============================")

if __name__ == "__main__":
    main()