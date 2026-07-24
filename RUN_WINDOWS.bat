@echo off
REM ====== Credit Default Predictor - Windows one-click runner ======
REM Double-click this file, or run it from Command Prompt.

echo.
echo [1/4] Creating a private Python environment...
python -m venv .venv
call .venv\Scripts\activate.bat

echo.
echo [2/4] Installing the required libraries (one-time, ~2 min)...
python -m pip install --upgrade pip >nul
python -m pip install -r requirements.txt

echo.
echo [3/4] Training the models (this can take a few minutes)...
python src\train.py

echo.
echo [4/4] Launching the web app in your browser...
streamlit run ui\streamlit_app.py

pause
