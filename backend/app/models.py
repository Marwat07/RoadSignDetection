from pydantic import BaseModel
from typing import List, Optional, Union
from enum import Enum

class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"

class DetectionResult(BaseModel):
    class_name: str
    class_id: int
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    frame_number: Optional[int] = None  # For video detections

class DetectionResponse(BaseModel):
    success: bool
    message: str
    media_type: MediaType
    detections: List[DetectionResult]
    total_detections: int
    processing_time: Optional[float] = None
    annotated_media: str  # Base64 encoded JPEG image or MP4 video with detections
    original_size: Optional[str] = None
    video_duration: Optional[float] = None
    frames_processed: Optional[int] = None

class VideoProcessingRequest(BaseModel):
    confidence_threshold: float = 0.5
    process_every_n_frames: int = 1
    output_fps: int = 10