@echo off
if not exist "%~dp0..\runtime\python.bat" (
    echo Isaac Sim is not linked. See docs\INSTALLATION.md.
    exit /b 1
)
call "%~dp0..\runtime\python.bat" "%~dp0..\simulation\build_scene.py" %*
exit /b %errorlevel%
