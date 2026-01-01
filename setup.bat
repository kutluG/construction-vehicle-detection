@echo off
echo ================================================
echo   Construction Site Detection - Setup Script
echo ================================================
echo.

echo [1/3] Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10 or higher.
    pause
    exit /b 1
)
echo.

echo [2/3] Installing required packages...
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.

echo [3/3] Checking model weights...
if exist "runs\detect\y11x_10ep\weights\best.pt" (
    echo ✓ Model weights found!
) else (
    echo ⚠ Model weights not found. Downloading from GitHub releases...
    echo.
    
    REM Create directory structure
    if not exist "runs\detect\y11x_10ep\weights" mkdir "runs\detect\y11x_10ep\weights"
    
    REM Download using PowerShell
    echo Downloading best.pt (109 MB)...
    powershell -Command "& {Invoke-WebRequest -Uri 'https://github.com/kutluG/construction-vehicle-detection/releases/download/v1.0.0/best.pt' -OutFile 'runs\detect\y11x_10ep\weights\best.pt'}"
    
    if exist "runs\detect\y11x_10ep\weights\best.pt" (
        echo ✓ Model downloaded successfully!
    ) else (
        echo ✗ Download failed. Please download manually from:
        echo https://github.com/kutluG/construction-vehicle-detection/releases/tag/v1.0.0
        pause
        exit /b 1
    )
)

echo.
echo ================================================
echo   Setup Complete!
echo ================================================
echo.
echo To start the web app, run: streamlit run app.py
echo Or double-click: run_app.bat
echo.
pause
