@echo off
"%~dp0..\.venv\Scripts\python.exe" "%~dp0..\tools\acquire_surroundocc_data.py" %*
exit /b %errorlevel%
