@echo off
REM Quick training script for large dataset

echo.
echo ===============================================
echo  HMM POS Tagger - Large Dataset Training
echo ===============================================
echo.

cd notebooks

echo Installing dependencies...
pip install tqdm nltk -q

echo.
echo Starting training on large dataset...
python train_large.py

if errorlevel 1 (
    echo.
    echo ERROR: Training failed!
    pause
    exit /b 1
)

echo.
echo ===============================================
echo Training completed successfully!
echo Model saved to: models\hmm_tagger_large.pkl
echo ===============================================
echo.
echo Next step: Run evaluate.bat to test the model
echo.
pause
