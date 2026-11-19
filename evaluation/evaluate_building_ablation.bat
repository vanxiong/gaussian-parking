@echo off
"%~dp0..\.venv\Scripts\python.exe" "%~dp0..\tools\run_building_ablation.py" %*
exit /b %errorlevel%
