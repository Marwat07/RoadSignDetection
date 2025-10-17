@echo off
echo Starting Road Sign Detection Frontend...
cd frontend
echo Installing dependencies...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install dependencies
    pause
    exit /b 1
)

echo Starting frontend server...
echo Make sure backend is running at: http://localhost:8000
echo Frontend will be available at: http://localhost:7862
set API_URL=http://localhost:8000
python app.py
pause
