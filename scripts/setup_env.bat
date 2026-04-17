@echo off
REM scripts\setup_env.bat
REM One-click environment setup for Windows

echo [1/4] Checking Python version...
python --version
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ from https://www.python.org
    pause
    exit /b 1
)

echo [2/4] Creating virtual environment...
if not exist venv (
    python -m venv venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists, skipping.
)

echo [3/4] Activating venv and upgrading pip...
call venv\Scripts\activate
python -m pip install --upgrade pip

echo [4/4] Installing dependencies...
pip install -r requirements.txt

echo.
echo ============================================
echo  Setup complete!
echo  Activate env with:  venv\Scripts\activate
echo ============================================
pause
