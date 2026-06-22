@echo off
REM Spousteci skript pro ACL Analyzu GUI
REM Tento skript spusti virtualni prostredi a nasledne pythonw.exe, 
REM aby nebylo zobrazeno prebytecne cerne terminalove okno.

cd /d "%~dp0"
start "" .venv\pythonw.exe Source\main\gui_app.py
exit
