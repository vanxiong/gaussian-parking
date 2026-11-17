@echo off
call "%~dp0tools\prepare_runtime.bat"
if errorlevel 1 exit /b 1
call "%~dp0runtime\python.bat" "%~dp0simulation\driving_scene.py" %*
if errorlevel 1 pause
