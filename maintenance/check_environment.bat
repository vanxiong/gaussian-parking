@echo off
setlocal
for %%I in ("%~dp0..") do set "TASK_ROOT=%%~fI"
for /f "delims=" %%P in ('wsl.exe --distribution Ubuntu-24.04 --exec wslpath -a "%TASK_ROOT%"') do set "LINUX_ROOT=%%P"
wsl.exe --distribution Ubuntu-24.04 --exec bash "%LINUX_ROOT%/tools/check_linux.sh"
exit /b %errorlevel%
