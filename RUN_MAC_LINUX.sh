#!/usr/bin/env bash
# ====== Credit Default Predictor - Mac/Linux one-click runner ======
# Run with:  bash RUN_MAC_LINUX.sh
set -e

echo ""
echo "[1/4] Creating a private Python environment..."
python3 -m venv .venv
source .venv/bin/activate

echo ""
echo "[2/4] Installing the required libraries (one-time, ~2 min)..."
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt

echo ""
echo "[3/4] Training the models (this can take a few minutes)..."
python src/train.py

echo ""
echo "[4/4] Launching the web app in your browser..."
streamlit run ui/streamlit_app.py
