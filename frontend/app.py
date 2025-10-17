import gradio as gr
import requests
import numpy as np
from PIL import Image
import io
import cv2
import os
import tempfile
import base64
from typing import Tuple, List

# Backend API URL
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

def detect_media(file, confidence_threshold=0.5, video_processing_mode="standard", process_every_n_frames=5):
    """
    Send image or video to backend API for road sign detection
    """
    try:
        # Determine if file is image or video
        filename = file.name.lower()
        is_video = any(filename.endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm'])
        
        # Prepare file for upload with proper resource management
        file_handle = open(file.name, 'rb')
        files = {'file': (os.path.basename(file.name), file_handle, 
                         'video/mp4' if is_video else 'image/jpeg')}
        
        params = {'confidence_threshold': confidence_threshold}
        
        # Choose appropriate API endpoint
        if is_video and video_processing_mode == "advanced":
            api_url = f"{API_URL}/detect-video-advanced"
            params.update({
                'process_every_n_frames': process_every_n_frames,
                'output_fps': 10
            })
        else:
            api_url = f"{API_URL}/detect"
        
        # Send request to backend with timeout and better error handling
        try:
            response = requests.post(api_url, files=files, params=params, timeout=300)  # 5 minute timeout
        except requests.exceptions.Timeout:
            return None, "Error: Request timed out. File may be too large or server is overloaded.", None
        except requests.exceptions.ConnectionError:
            return None, "Error: Cannot connect to backend server. Please ensure the backend is running.", None
        except requests.exceptions.RequestException as e:
            return None, f"Error: Request failed - {str(e)}", None
        
        if response.status_code == 200:
            try:
                result = response.json()
            except requests.exceptions.JSONDecodeError:
                return None, "Error: Invalid JSON response from backend", None
            
            if result['media_type'] == 'image':
                # Convert annotated image bytes back to PIL Image
                if isinstance(result['annotated_media'], str):
                    # Base64 encoded
                    import base64
                    annotated_bytes = io.BytesIO(base64.b64decode(result['annotated_media']))
                else:
                    # Raw bytes
                    annotated_bytes = io.BytesIO(bytes(result['annotated_media']))
                annotated_image = Image.open(annotated_bytes)
                
                # Create detection summary
                detection_summary = create_detection_summary(result)
                
                return annotated_image, detection_summary, None
                
            else:  # video
                # Handle video from backend - save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
                    if isinstance(result['annotated_media'], str):
                        if result['annotated_media'].startswith('/static/'):
                            # This is a URL - return error message instead to avoid SSRF
                            return None, "Video too large - please try with a smaller video or use batch processing", None
                        else:
                            # Base64 encoded video data
                            try:
                                video_data = base64.b64decode(result['annotated_media'])
                                f.write(video_data)
                            except Exception as e:
                                return None, f"Error decoding video data: {str(e)}", None
                    else:
                        # Raw bytes
                        f.write(bytes(result['annotated_media']))
                    
                    temp_video_path = f.name
                
                # Create detection summary for video
                detection_summary = create_video_detection_summary(result)
                
                return None, detection_summary, temp_video_path
                
        else:
            try:
                error_detail = response.json().get('detail', f'HTTP {response.status_code}')
                error_msg = f"Error: {error_detail}"
            except:
                error_msg = f"Error: HTTP {response.status_code} - {response.text[:200]}"
            return None, error_msg, None
    
    except Exception as e:
        error_msg = f"Error processing media: {str(e)}"
        return None, error_msg, None
    finally:
        # Close file handle if it was opened
        try:
            if 'file_handle' in locals():
                file_handle.close()
        except:
            pass

def create_detection_summary(result):
    """Create detection summary for image results"""
    detection_summary = f"✅ **Detection Complete**\n\n"
    detection_summary += f"**Media Type:** {result['media_type'].upper()}\n"
    detection_summary += f"**Original Size:** {result['original_size']}\n"
    detection_summary += f"**Processing Time:** {result['processing_time']:.2f}s\n"
    detection_summary += f"**Total Detections:** {result['total_detections']}\n\n"
    
    if result['detections']:
        detection_summary += "**Detected Road Signs:**\n\n"
        for i, detection in enumerate(result['detections']):
            detection_summary += (
                f"{i+1}. **{detection['class_name']}** "
                f"(Confidence: {detection['confidence']:.2f})\n"
                f"   Location: {[int(x) for x in detection['bbox']]}\n\n"
            )
    else:
        detection_summary += "**No road signs detected**\n"
    
    return detection_summary

def create_video_detection_summary(result):
    """Create detection summary for video results"""
    detection_summary = f"🎥 **Video Processing Complete**\n\n"
    detection_summary += f"**Original Size:** {result['original_size']}\n"
    detection_summary += f"**Video Duration:** {result['video_duration']:.2f}s\n"
    detection_summary += f"**Frames Processed:** {result['frames_processed']}\n"
    detection_summary += f"**Processing Time:** {result['processing_time']:.2f}s\n"
    detection_summary += f"**Total Detections:** {result['total_detections']}\n\n"
    
    if result['detections']:
        # Group detections by class
        class_counts = {}
        for detection in result['detections']:
            class_name = detection['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        detection_summary += "**Detection Summary by Class:**\n\n"
        for class_name, count in class_counts.items():
            detection_summary += f"• **{class_name}**: {count} detections\n"
        
        # Show some frame-by-frame details
        detection_summary += f"\n**First few detections:**\n\n"
        for i, detection in enumerate(result['detections'][:5]):
            detection_summary += (
                f"Frame {detection.get('frame_number', 'N/A')}: "
                f"{detection['class_name']} ({detection['confidence']:.2f})\n"
            )
    else:
        detection_summary += "**No road signs detected in the video**\n"
    
    return detection_summary

def create_demo():
    """
    Create Gradio interface with image and video support
    """
    with gr.Blocks(
        title="Road Sign Detection System",
        theme=gr.themes.Soft(),
        css="""
        .video-container { max-width: 100%; margin: auto; }
        .summary-box { max-height: 500px; overflow-y: auto; }
        """
    ) as demo:
        
        gr.Markdown(
            """
            # 🚦 Road Sign Detection System
            **Detect and classify road signs in Images and Videos using YOLOv8**
            
            Upload an image or video containing road signs to see the detection results.
            """
        )
        
        with gr.Tab("Single Media Detection"):
            with gr.Row():
                with gr.Column():
                    media_input = gr.File(
                        label="Upload Image or Video",
                        file_types=[".png", ".jpg", ".jpeg", ".mp4", ".avi", ".mov"],
                        type="filepath"
                    )
                    
                    with gr.Accordion("Detection Settings", open=False):
                        confidence_slider = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.5,
                            step=0.1,
                            label="Confidence Threshold"
                        )
                        
                        video_mode = gr.Radio(
                            choices=["standard", "advanced"],
                            value="standard",
                            label="Video Processing Mode"
                        )
                        
                        frame_sampling = gr.Slider(
                            minimum=1,
                            maximum=30,
                            value=5,
                            step=1,
                            label="Process Every N Frames (Advanced Mode)",
                            visible=False
                        )
                    
                    detect_btn = gr.Button("Detect Road Signs", variant="primary", size="lg")
                
                with gr.Column():
                    image_output = gr.Image(
                        label="Detection Results (Image)",
                        type="pil",
                        visible=True
                    )
                    
                    video_output = gr.Video(
                        label="Detection Results (Video)",
                        visible=False
                    )
                    
                    summary_output = gr.Markdown(
                        label="Detection Summary",
                        elem_classes=["summary-box"]
                    )
            
            # Show/hide video settings based on mode
            def toggle_video_settings(mode):
                return gr.update(visible=(mode == "advanced"))
            
            video_mode.change(
                toggle_video_settings,
                inputs=[video_mode],
                outputs=[frame_sampling]
            )
            
            # Update output visibility based on media type
            def update_output_visibility(file_path):
                if file_path is None:
                    return gr.update(visible=True), gr.update(visible=False)
                
                filename = file_path.lower()
                is_video = any(filename.endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv'])
                
                if is_video:
                    return gr.update(visible=False), gr.update(visible=True)
                else:
                    return gr.update(visible=True), gr.update(visible=False)
            
            media_input.change(
                update_output_visibility,
                inputs=[media_input],
                outputs=[image_output, video_output]
            )
            
            detect_btn.click(
                fn=detect_media,
                inputs=[media_input, confidence_slider, video_mode, frame_sampling],
                outputs=[image_output, summary_output, video_output]
            )
        
        with gr.Tab("Batch Processing"):
            with gr.Row():
                with gr.Column():
                    batch_files = gr.File(
                        file_count="multiple",
                        file_types=[".png", ".jpg", ".jpeg", ".mp4", ".avi", ".mov"],
                        label="Upload Multiple Images/Videos"
                    )
                    batch_confidence = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.5,
                        step=0.1,
                        label="Confidence Threshold"
                    )
                    batch_detect_btn = gr.Button("Process Batch", variant="primary")
                
                with gr.Column():
                    batch_results = gr.Gallery(
                        label="Image Results",
                        show_label=True,
                        elem_id="gallery"
                    )
                    batch_videos = gr.File(
                        label="Video Results",
                        file_count="multiple",
                        type="filepath"
                    )
                    batch_summary = gr.JSON(
                        label="Batch Processing Summary"
                    )
            
            batch_detect_btn.click(
                fn=process_batch_detection,
                inputs=[batch_files, batch_confidence],
                outputs=[batch_results, batch_videos, batch_summary]
            )
        
        with gr.Tab("Real-time Demo"):
            gr.Markdown(
                """
                ## Real-time Detection Demo
                
                For real-time detection, you would need to:
                1. Use a webcam feed
                2. Stream processing through the API
                3. Display results in real-time
                
                *This feature requires additional setup for streaming and is not implemented in this version.*
                
                ### Alternative Options:
                - Record a video with your camera and upload it using the main detection tab
                - Use video capture tools to save camera feed then process
                """
            )
        
        with gr.Tab("About & Instructions"):
            gr.Markdown(
                """
                ## 📖 Instructions
                
                ### Supported Media:
                - **Images**: JPG, JPEG, PNG, BMP
                - **Videos**: MP4, AVI, MOV, MKV, WebM
                
                ### Processing Modes:
                - **Standard**: Processes every frame (slower but more accurate)
                - **Advanced**: Configurable frame sampling (faster processing)
                
                ### Detection Settings:
                - **Confidence Threshold**: Adjust detection sensitivity
                - **Frame Sampling**: Process every N frames (video only)
                
                ## 🛠️ Technical Details
                
                This system uses:
                - **Backend**: FastAPI with YOLOv8 model
                - **Frontend**: Gradio interface
                - **Model**: Custom-trained YOLOv8 on road sign dataset
                
                ## 📊 Performance Tips
                - For videos longer than 30 seconds, use Advanced mode
                - Adjust confidence threshold based on your needs
                - Higher frame sampling = faster processing but may miss detections
                """
            )
        
        # Note: Examples section removed as example files are not provided
        # You can add your own examples by placing files in the frontend directory
        # and uncommenting the Examples section
    
    return demo

def process_batch_detection(files, confidence_threshold):
    """Process multiple files in batch"""
    image_results = []
    video_results = []
    processing_summary = []
    
    for file in files:
        try:
            image_output, summary, video_output = detect_media(file, confidence_threshold)
            
            if image_output is not None:
                image_results.append(image_output)
            if video_output is not None:
                video_results.append(video_output)
            
            processing_summary.append({
                "filename": os.path.basename(file.name),
                "status": "success",
                "summary": summary
            })
            
        except Exception as e:
            processing_summary.append({
                "filename": os.path.basename(file.name),
                "status": "error",
                "error": str(e)
            })
    
    return image_results, video_results, processing_summary

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7862,
        share=False,
        show_error=True,
        allowed_paths=["."]
    )
