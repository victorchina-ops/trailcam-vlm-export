@echo off
rem Trail-camera analysis with a local vision-language model (Ollama). Pass a folder or options, e.g.
rem   run_vlm.cmd "D:\Trail Photos\Camera 1" --limit 20
rem run_vlm_gpu.cmd and run_vlm_cpu.cmd call this file with --profile gpu or --profile cpu.
setlocal EnableDelayedExpansion
rem Delayed expansion keeps characters in the launch command literal; Python uses it to pause a double-clicked window.
set "TRAILCAM_VLM_LAUNCH=!CMDCMDLINE!"
setlocal DisableDelayedExpansion
pushd "%~dp0" 2>nul
if errorlevel 1 goto :no_folder
rem The marker is written only after a complete installation, so an interrupted first run installs again.
if exist ".venv\installed.ok" goto :run
echo Setting up the Python environment (first run only)...
if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv .venv 2>nul || py -3.11 -m venv .venv 2>nul || python -m venv .venv
)
if not exist ".venv\Scripts\python.exe" goto :no_python
".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt
if errorlevel 1 goto :install_failed
echo ok> ".venv\installed.ok"
:run
".venv\Scripts\python.exe" -m trailcam_vlm %*
rem %errorlevel% is expanded before popd runs, so the exit code is Python's.
popd & exit /b %errorlevel%

:no_folder
echo Cannot open the folder of this launcher. Copy the tool folder to a local drive.
goto :failed
:no_python
echo Could not create a Python environment. Install 64-bit Python 3.12 and run this file again:
echo   winget install -e --id Python.Python.3.12
goto :failed
:install_failed
echo Installing the Python packages failed. Check the internet connection and run this file again.
:failed
popd 2>nul
setlocal EnableDelayedExpansion
if /i not "!TRAILCAM_VLM_LAUNCH:/c =!"=="!TRAILCAM_VLM_LAUNCH!" pause
exit /b 2
