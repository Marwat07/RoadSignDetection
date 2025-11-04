# Road Sign Detection System
<img width="1892" height="998" alt="image" src="https://github.com/user-attachments/assets/bf6f19fb-3eb5-41f1-b17b-45900a867ede" />


A YOLOv8-based road sign detection system with FastAPI backend and Gradio frontend that supports both images and videos.

## Features

- 🚦 Detect and classify road signs in images and videos
- 🎯 Real-time confidence scoring
- 🎬 Video processing with frame-by-frame analysis
- 🌐 Web interface using Gradio
- 🚀 FastAPI REST API backend
- 🐳 Docker support (optional)

## Quick Start (Without Docker)

### Prerequisites

- Python 3.9+ (tested with Python 3.13.2)
- Windows 10/11 (scripts provided for Windows)

### Setup Instructions

1. **Install Backend Dependencies and Start Server:**
   ```bash
   # Double-click run_backend.bat or run in command prompt:
   run_backend.bat
   ```

2. **Install Frontend Dependencies and Start Interface:**
   ```bash
   # In a new terminal, double-click run_frontend.bat or run:
   run_frontend.bat
   ```

3. **Access the Application:**
   - Backend API: http://localhost:8000
   - Frontend Interface: http://localhost:7860
   - API Documentation: http://localhost:8000/docs

## Manual Setup

### Backend Setup

```bash
cd backend
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host localhost --port 8000 --reload
```

### Frontend Setup

```bash
cd frontend
python -m pip install -r requirements.txt
set API_URL=http://localhost:8000
python app.py
```

## Docker Setup (Optional)

If you have Docker installed:

```bash
docker-compose up --build
```

## Troubleshooting

### Common Issues

1. **Import Errors:**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility

2. **Model File Missing:**
   - Ensure `backend/models/best.pt` exists
   - Download or train a YOLOv8 model if missing

3. **Port Already in Use:**
   - Backend: Change port in `run_backend.bat` from 8000 to another port
   - Frontend: Modify `app.py` to use a different port

4. **API Connection Issues:**
   - Ensure backend is running before starting frontend
   - Check that API_URL in frontend matches backend address
