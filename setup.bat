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
if exist "runs\detect\y11s_all_vehicles\weights\best.pt" (
    echo ✓ Model weights found!
) else (
    echo.
    echo ⚠ WARNING: Model weights not found!
    echo Please download best.pt and place it in:
    echo runs\detect\y11s_all_vehicles\weights\best.pt
    echo.
    echo You can get it from:
    echo - GitHub Releases page
    echo - Or train your own model with: python train.py
    echo.
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
