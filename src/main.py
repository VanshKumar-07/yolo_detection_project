import cv2
import os
from ultralytics import YOLO

def process_video():
    print("--- Starting YOLO Object Detection Pipeline ---")

    # 1. Setup bulletproof dynamic paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_video_path = os.path.normpath(os.path.join(current_dir, "../data/sample_video.mp4"))
    output_video_path = os.path.normpath(os.path.join(current_dir, "../outputs/processed_video.mp4"))

    if not os.path.exists(input_video_path):
        print(f"[ERROR] Cannot find video at {input_video_path}")
        print("Please place a file named 'sample_video.mp4' in the data folder.")
        return

    # 2. Load the YOLOv8 model (this will automatically download the lightweight 'nano' model on first run)
    print("[INFO] Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt") 

    # 3. Initialize Video Capture
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("[ERROR] Error opening video stream or file")
        return

    # Get video properties to match the output format
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 4. Initialize Video Writer
    # We use mp4v codec for standard mp4 saving
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    print(f"[INFO] Processing video... This may take a few minutes depending on your hardware.")
    frame_count = 0

    # 5. Process frame-by-frame
    while cap.isOpened():
        success, frame = cap.read()
        
        if not success:
            break # End of video reached
            
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"   -> Processing frame {frame_count}")

        # Run YOLO inference on the frame
        # conf=0.5 means it only draws a box if it is 50% sure about the object
        results = model(frame, conf=0.5, verbose=False)

        # The model returns a list of results. We plot the bounding boxes directly onto the frame
        annotated_frame = results[0].plot()

        # Write the annotated frame to our new output video
        out.write(annotated_frame)

    # 6. Clean up resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"[SUCCESS] Finished processing {frame_count} frames!")
    print(f"[SUCCESS] Saved to {output_video_path}")

if __name__ == "__main__":
    process_video()