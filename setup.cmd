@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" py -3 -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" prepare_model.py
if errorlevel 1 goto failed
echo Ready. Open start.cmd to launch.
pause
exit /b 0
:failed
echo Setup failed. Check the error above and retry.
pause
exit /b 1
