@echo off
REM Quick evaluation script

echo.
echo ===============================================
echo  HMM POS Tagger - Model Evaluation
echo ===============================================
echo.

cd notebooks

echo Installing dependencies...
pip install tqdm nltk -q

echo.
echo Starting evaluation...
python evaluate_large.py

if errorlevel 1 (
    echo.
    echo ERROR: Evaluation failed!
    pause
    exit /b 1
)

echo.
echo ===============================================
echo Evaluation completed!
echo ===============================================
echo.
pause
