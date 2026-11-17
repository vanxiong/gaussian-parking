@echo off
setlocal
for %%I in ("%~dp0..") do set "TASK_ROOT=%%~fI"
for /f "delims=" %%P in ('wsl.exe --distribution Ubuntu-24.04 --exec wslpath -a "%TASK_ROOT%"') do set "LINUX_ROOT=%%P"
wsl.exe --distribution Ubuntu-24.04 --exec bash "%LINUX_ROOT%/tools/build_runtime.sh"
if errorlevel 1 exit /b 1
wsl.exe --distribution Ubuntu-24.04 --user root --exec bash "%LINUX_ROOT%/tools/mount_runtime.sh"
exit /b %errorlevel%
