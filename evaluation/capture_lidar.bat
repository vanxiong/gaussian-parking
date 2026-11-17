@echo off
call "%~dp0..\runtime\python.bat" "%~dp0..\simulation\capture_cameras.py" --lidar %*
exit /b %errorlevel%
