import cv2
import numpy as np
from PIL import Image
import io
import os
import tempfile
from typing import Tuple, List
from .models import DetectionResult
import time

def process_image(image: Image.Image) -> np.ndarray:
    """Process image for YOLOv8 inference"""
    return np.array(image)

def draw_detections(image: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
    """Draw bounding boxes and labels on image"""
    image_copy = image.copy()
    
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection.bbox)
        confidence = detection.confidence
        class_name = detection.class_name
        
        # Draw bounding box
        color = (0, 255, 0)  # Green
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(image_copy, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(image_copy, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return image_copy

def process_video_detection(frame: np.ndarray, model, confidence_threshold: float) -> Tuple[np.ndarray, List[DetectionResult]]:
    """Process a single video frame for detection"""
    detections = []
    
    # Run YOLOv8 inference
    results = model(frame, conf=confidence_threshold)
    result = results[0]
    
    if result.boxes is not None:
        for box in result.boxes:
            detection = DetectionResult(
                class_name=model.names[int(box.cls)],
                class_id=int(box.cls),
                confidence=float(box.conf),
                bbox=[
                    float(box.xyxy[0][0]),  # x1
                    float(box.xyxy[0][1]),  # y1
                    float(box.xyxy[0][2]),  # x2
                    float(box.xyxy[0][3])   # y2
                ]
            )
            detections.append(detection)
    
    # Draw detections on frame
    annotated_frame = draw_detections(frame, detections)
    
    return annotated_frame, detections

def create_video_from_frames(frames: List[np.ndarray], output_path: str, fps: int = 10):
    """Create video from list of frames"""
    if not frames:
        raise ValueError("No frames to process")
    
    height, width = frames[0].shape[:2]
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()

def get_media_type(filename: str) -> str:
    """Determine if file is image or video based on extension"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
    
    ext = os.path.splitext(filename.lower())[1]
    
    if ext in image_extensions:
        return "image"
    elif ext in video_extensions:
        return "video"
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def read_video_bytes(video_path: str) -> bytes:
    """Read video file as bytes"""
    with open(video_path, 'rb') as f:
        return f.read()