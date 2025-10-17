@echo off
echo Starting Road Sign Detection System...
echo.

echo Starting Backend Server...
start "Backend Server" cmd /c "run_backend.bat"

echo Waiting for backend to start...
timeout /t 5 /nobreak > nul

echo Starting Frontend Interface...
start "Frontend Interface" cmd /c "run_frontend.bat"

echo.
echo ===============================================
echo  Road Sign Detection System Starting...
echo ===============================================
echo.
echo Backend API: http://localhost:8000
echo Frontend UI: http://localhost:7862
echo API Docs: http://localhost:8000/docs
echo.
echo Both services are starting in separate windows.
echo Wait a few seconds, then open your browser to:
echo http://localhost:7862
echo.
pause