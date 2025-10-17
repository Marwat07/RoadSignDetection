from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import cv2 
import numpy as np
from PIL import Image
import io
from ultralytics import YOLO 
import json
from datetime import datetime
import time
import os
import tempfile
import aiofiles
import asyncio
from typing import List
import base64

from .models import DetectionResponse, DetectionResult, MediaType, VideoProcessingRequest
from .utils import (
    process_image, draw_detections, process_video_detection, 
    create_video_from_frames, get_media_type, read_video_bytes
)

app = FastAPI(
    title="Road Sign Detection API",
    description="YOLOv8 based Road Sign Detection System with Image and Video support",
    version="2.0.0"
)

# Configure max upload size (100MB)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
)

# Add middleware to handle large file uploads
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        print(f"Middleware error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error occurred"}
        )


# Create temp directory
os.makedirs("temp_uploads", exist_ok=True)
os.makedirs("temp_results", exist_ok=True)

# Mount static files for video serving
app.mount("/static", StaticFiles(directory="temp_results"), name="static")

# Load YOLOv8 model
model_path = os.path.join("models", "best.pt")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = YOLO(model_path)

@app.get("/")
async def root():
    return {"message": "Road Sign Detection API", "version": "2.0.0", "support": "images and videos"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/test-static")
async def test_static():
    """Test endpoint to verify static file serving is working"""
    return {"message": "Static file serving configured", "static_path": "/static/"}

@app.post("/detect", response_model=DetectionResponse)
async def detect_roadsigns(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5
):
    """
    Detect road signs in uploaded image or video
    """
    start_time = time.time()
    
    try:
        # Determine media type
        media_type = get_media_type(file.filename)
        
        # Read file content
        content = await file.read()
        
        if media_type == MediaType.IMAGE:
            return await process_image_detection(content, file.filename, confidence_threshold, start_time)
        else:
            return await process_video_detection_simple(content, file.filename, confidence_threshold, start_time)
            
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        print(f"Unexpected error in detect_roadsigns: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

async def process_image_detection(content: bytes, filename: str, confidence_threshold: float, start_time: float):
    """Process image detection"""
    # Process image
    image = Image.open(io.BytesIO(content))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    processed_image = process_image(image)
    
    # Run YOLOv8 inference
    results = model(processed_image, conf=confidence_threshold)
    
    # Parse results
    detections = []
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
    
    # Create annotated image
    annotated_image = draw_detections(np.array(image), detections)
    _, encoded_image = cv2.imencode('.jpg', annotated_image)
    encoded_image_bytes = encoded_image.tobytes()
    
    # Convert to base64 for JSON serialization
    annotated_media_b64 = base64.b64encode(encoded_image_bytes).decode('utf-8')
    
    processing_time = time.time() - start_time
    
    response = DetectionResponse(
        success=True,
        message=f"Detected {len(detections)} road signs",
        media_type=MediaType.IMAGE,
        detections=detections,
        total_detections=len(detections),
        processing_time=processing_time,
        annotated_media=annotated_media_b64,
        original_size=f"{image.width}x{image.height}"
    )
    
    return response

async def process_video_detection_simple(content: bytes, filename: str, confidence_threshold: float, start_time: float):
    """Process video detection with simple approach"""
    # Save uploaded video to temp file
    temp_input_path = f"temp_uploads/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
    temp_output_path = f"temp_results/{datetime.now().strftime('%Y%m%d_%H%M%S')}_output_{filename}"
    
    try:
        # Save uploaded video
        async with aiofiles.open(temp_input_path, 'wb') as f:
            await f.write(content)
        
        # Open video
        cap = cv2.VideoCapture(temp_input_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video file")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Prepare video writer with web-compatible codec
        try:
            fourcc = cv2.VideoWriter_fourcc(*'H264')  # Try H264 first
            out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
            if not out.isOpened():
                raise Exception("H264 codec failed")
        except:
            # Fallback to MP4V if H264 fails
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
        
        processed_frames = 0
        all_detections = []
        frame_number = 0
        
        # Process video frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every frame
            annotated_frame, frame_detections = process_video_detection(frame, model, confidence_threshold)
            
            # Add frame number to detections
            for detection in frame_detections:
                detection.frame_number = frame_number
                all_detections.append(detection)
            
            out.write(annotated_frame)
            processed_frames += 1
            frame_number += 1
        
        # Release resources
        cap.release()
        out.release()
        
        # Read output video as bytes
        output_video_bytes = read_video_bytes(temp_output_path)
        
        # For small videos, use base64. For large videos, create a download link
        video_size_mb = len(output_video_bytes) / (1024 * 1024)
        
        if video_size_mb < 50:  # Less than 50MB, use base64
            annotated_media_b64 = base64.b64encode(output_video_bytes).decode('utf-8')
        else:
            # For large videos, save to static directory
            import uuid
            unique_filename = f"video_{uuid.uuid4().hex[:8]}.mp4"
            static_video_path = os.path.join("temp_results", unique_filename)
            
            import shutil
            shutil.copy2(temp_output_path, static_video_path)
            
            annotated_media_b64 = f"/static/{unique_filename}"
        
        processing_time = time.time() - start_time
        
        response = DetectionResponse(
            success=True,
            message=f"Processed {processed_frames} frames, detected {len(all_detections)} road signs",
            media_type=MediaType.VIDEO,
            detections=all_detections,
            total_detections=len(all_detections),
            processing_time=processing_time,
            annotated_media=annotated_media_b64,
            original_size=f"{width}x{height}",
            video_duration=duration,
            frames_processed=processed_frames
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    finally:
        # Cleanup temp files
        for temp_file in [temp_input_path, temp_output_path]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

@app.post("/detect-video-advanced")
async def detect_video_advanced(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5,
    process_every_n_frames: int = 1,
    output_fps: int = 10
):
    """
    Advanced video detection with configurable frame sampling
    """
    start_time = time.time()
    
    try:
        if get_media_type(file.filename) != MediaType.VIDEO:
            raise HTTPException(status_code=400, detail="File must be a video")
        
        content = await file.read()
        
        # Save uploaded video to temp file
        temp_input_path = f"temp_uploads/advanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        temp_output_path = f"temp_results/advanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}_output_{file.filename}"
        
        async with aiofiles.open(temp_input_path, 'wb') as f:
            await f.write(content)
        
        # Process video
        cap = cv2.VideoCapture(temp_input_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video file")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Prepare video writer with configured FPS and web-compatible codec
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(temp_output_path, fourcc, output_fps, (width, height))
        
        processed_frames = 0
        all_detections = []
        frame_number = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame based on sampling rate
            if frame_number % process_every_n_frames == 0:
                annotated_frame, frame_detections = process_video_detection(frame, model, confidence_threshold)
                
                for detection in frame_detections:
                    detection.frame_number = frame_number
                    all_detections.append(detection)
                
                out.write(annotated_frame)
                processed_frames += 1
            
            frame_number += 1
        
        cap.release()
        out.release()
        
        # Read output video
        output_video_bytes = read_video_bytes(temp_output_path)
        
        processing_time = time.time() - start_time
        
        response = DetectionResponse(
            success=True,
            message=f"Processed {processed_frames} frames (sampling: 1/{process_every_n_frames}), detected {len(all_detections)} road signs",
            media_type=MediaType.VIDEO,
            detections=all_detections,
            total_detections=len(all_detections),
            processing_time=processing_time,
            annotated_media=output_video_bytes,
            original_size=f"{width}x{height}",
            video_duration=frame_number / fps if fps > 0 else 0,
            frames_processed=processed_frames
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.post("/detect-batch")
async def detect_batch(files: List[UploadFile] = File(...)):
    """Detect road signs in multiple images/videos"""
    results = []
    for file in files:
        try:
            detection_result = await detect_roadsigns(file)
            results.append({
                "filename": file.filename,
                "media_type": detection_result.media_type,
                "result": detection_result.dict()
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)