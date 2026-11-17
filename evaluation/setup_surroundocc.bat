@echo off
setlocal
for %%I in ("%~dp0..") do set "TASK_ROOT=%%~fI"
call "%TASK_ROOT%\tools\prepare_runtime.bat"
if errorlevel 1 exit /b 1
for /f "delims=" %%P in ('wsl.exe --distribution Ubuntu-24.04 --exec wslpath -a "%TASK_ROOT%"') do set "LINUX_ROOT=%%P"
wsl.exe --distribution Ubuntu-24.04 --exec bash "%LINUX_ROOT%/tools/setup_surroundocc.sh"
exit /b %errorlevel%
