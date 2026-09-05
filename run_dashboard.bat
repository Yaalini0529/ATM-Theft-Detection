@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo Creating Python environment...
    py -m venv venv
    if errorlevel 1 (
        echo Could not create the virtual environment. Install Python 3.11 or newer and try again.
        pause
        exit /b 1
    )
)

echo Installing required packages...
call "venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Dependency installation failed.
    pause
    exit /b 1
)

echo Starting ATM Security dashboard...
call "venv\Scripts\python.exe" -m streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
pause