@echo off
cd /d "%~dp0"
if errorlevel 1 goto failed
if not exist "%~dp0requirements.txt" goto incomplete
if not exist "%~dp0prepare_model.py" goto incomplete
if not exist "%~dp0app.py" goto incomplete
if not exist ".venv\Scripts\python.exe" py -3 -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" prepare_model.py
if errorlevel 1 goto failed
".venv\Scripts\python.exe" prepare_knowledge_model.py
if errorlevel 1 goto failed
echo Ready. Open start.cmd to launch.
pause
exit /b 0
:incomplete
echo Installation files are missing from this folder.
echo Extract ALL files from the downloaded ZIP first.
echo Do not run setup.cmd directly inside the ZIP or copy it out alone.
echo Open the extracted project folder, then run setup.cmd again.
pause
exit /b 1
:failed
echo Setup failed. Check the error above and retry.
pause
exit /b 1
