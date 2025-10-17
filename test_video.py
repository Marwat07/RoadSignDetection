import cv2
import numpy as np
import os

# Create a simple test video with some basic shapes
def create_test_video():
    output_path = "test_video.mp4"
    
    # Video properties
    width, height = 640, 480
    fps = 30
    duration = 3  # seconds
    total_frames = fps * duration
    
    # Try different codecs
    codecs = ['mp4v', 'XVID', 'MJPG']
    
    for codec in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                print(f"Failed to open video writer with {codec}")
                continue
                
            print(f"Creating test video with {codec} codec...")
            
            for frame_num in range(total_frames):
                # Create a frame with moving rectangle
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Background
                frame[:] = (50, 50, 50)
                
                # Moving rectangle (simulating a road sign)
                x = int(100 + (frame_num * 2) % 400)
                y = int(height // 2 - 50)
                
                cv2.rectangle(frame, (x, y), (x + 100, y + 100), (0, 255, 0), -1)
                cv2.putText(frame, "TEST SIGN", (x + 10, y + 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                out.write(frame)
            
            out.release()
            
            if os.path.exists(output_path):
                print(f"Successfully created test video: {output_path}")
                print(f"File size: {os.path.getsize(output_path)} bytes")
                return True
            else:
                print(f"Failed to create video with {codec}")
                
        except Exception as e:
            print(f"Error with {codec}: {e}")
            continue
    
    return False

if __name__ == "__main__":
    if create_test_video():
        print("Test video created successfully!")
    else:
        print("Failed to create test video")