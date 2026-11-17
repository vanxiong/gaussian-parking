@echo off
rem Mount project-local dependencies; no ROS services are started.
setlocal
for %%I in ("%~dp0..") do set "TASK_ROOT=%%~fI"
if not exist "%TASK_ROOT%\runtime\python.bat" (
    echo Isaac Sim is not linked. See docs\INSTALLATION.md.
    echo Run maintenance\link_isaac_sim.ps1 with your Isaac Sim installation path.
    exit /b 1
)
if not exist "%TASK_ROOT%\.runtime-linux\dependencies.squashfs" exit /b 0
if "%GF3D_FAST_RUNTIME%"=="0" exit /b 0
for /f "delims=" %%P in ('wsl.exe --distribution Ubuntu-24.04 --exec wslpath -a "%TASK_ROOT%"') do set "LINUX_ROOT=%%P"
if not defined LINUX_ROOT exit /b 1
wsl.exe --distribution Ubuntu-24.04 --user root --exec bash "%LINUX_ROOT%/tools/mount_runtime.sh"
exit /b %errorlevel%
