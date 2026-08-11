@echo off
cd /d "%~dp0"

REM Prefer the known new_gello env; fall back to conda activate.
set "PY=F:\conda_envs\new_gello\python.exe"
if exist "%PY%" (
  "%PY%" gello_tianji_teleop_gui.py %*
) else (
  call conda activate new_gello
  python gello_tianji_teleop_gui.py %*
)
if errorlevel 1 pause
