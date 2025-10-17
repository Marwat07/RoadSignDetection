@echo off
echo Starting Road Sign Detection Backend...
cd backend
echo Installing dependencies...
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install dependencies
    pause
    exit /b 1
)

echo Starting backend server...
echo Backend API will be available at: http://localhost:8000
echo API Documentation available at: http://localhost:8000/docs
python -m uvicorn app.main:app --host localhost --port 8000 --reload
pause
