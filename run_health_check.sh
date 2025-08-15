#!/bin/bash

echo "CSV Data Analyst - Health Check"
echo "================================"

# Check if virtual environment exists
if [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Warning: Virtual environment not found at venv/bin/activate"
    echo "Continuing with system Python..."
fi

echo ""
echo "Running health check..."
python health_check.py

echo ""
read -p "Press Enter to continue..."