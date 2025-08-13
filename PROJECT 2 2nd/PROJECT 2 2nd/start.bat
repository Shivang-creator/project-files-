@echo off
echo Starting Data Analyst Agent...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist .env (
    echo Warning: .env file not found
    echo Creating .env file from template...
    copy .env.example .env >nul 2>&1
    echo.
    echo Please edit .env file and add your API keys, then run this script again.
    echo.
    pause
    exit /b 1
)

REM Install dependencies
echo Installing dependencies...
pip install -r requirements_simple.txt

if errorlevel 1 (
    echo.
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Starting server...
echo Access the web interface at: http://localhost:8000/frontend.html
echo API documentation at: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

python main.py
