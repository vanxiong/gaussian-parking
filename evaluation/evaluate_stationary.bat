@echo off
"%~dp0..\.venv\Scripts\python.exe" "%~dp0..\tools\run_occupancy_evaluation.py" --stationary %*
exit /b %errorlevel%
