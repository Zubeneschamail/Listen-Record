@echo off
cd /d "%~dp0"
if not exist "%~dp0app.py" (
  echo Extract ALL files from the ZIP, then run setup.cmd in that folder.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\pythonw.exe" (
  echo Please run setup.cmd first.
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" "app.py"
