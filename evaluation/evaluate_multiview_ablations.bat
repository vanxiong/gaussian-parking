@echo off
"%~dp0..\.venv\Scripts\python.exe" "%~dp0..\tools\run_multiview_ablations.py" %*
exit /b %errorlevel%
