@echo off
echo CSV Data Analyst - Health Check
echo ================================

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found at venv\Scripts\activate.bat
    echo Continuing with system Python...
)

echo.
echo Running health check...
python health_check.py

echo.
pause