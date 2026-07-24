# HOW TO RUN THIS — read me first (no experience needed)

This project predicts whether a credit card customer will miss their next payment.
Everything is already built and the models are already trained. You just need to run it.

---

## The absolute easiest way (one click)

**On Windows:** double-click the file **`RUN_WINDOWS.bat`**.
**On Mac/Linux:** open Terminal in this folder and type: `bash RUN_MAC_LINUX.sh` then press Enter.

That's it. It will set everything up and open the app in your web browser. The first
run takes a few minutes (it installs libraries and trains the models). Future runs are fast.

> If double-clicking the .bat does nothing, follow the manual steps below instead.

---

## Before you start: do you have Python?

This needs **Python 3.10 or newer**. To check:

1. Open a terminal:
   - **Windows:** press the Windows key, type `cmd`, press Enter.
   - **Mac:** press Cmd+Space, type `terminal`, press Enter.
2. Type this and press Enter:  `python --version`  (on Mac, try `python3 --version`)
3. If you see a number like `Python 3.11.x`, you're good.
   If you see an error, install Python from https://www.python.org/downloads/
   (on Windows, **tick the box "Add Python to PATH"** during install).

---

## Manual steps (if the one-click file didn't work)

Open a terminal **inside this project folder**. (Tip: in the folder, Windows users can
type `cmd` in the address bar and press Enter to open a terminal already in the right place.)

Then run these commands one at a time, pressing Enter after each:

**1. Install the libraries** (one time only):
```
pip install -r requirements.txt
```

**2. Train the models** (downloads data + trains; takes a few minutes):
```
python src/train.py
```
On Mac, if `python` doesn't work, use `python3` instead.

**3. Start the app:**
```
streamlit run ui/streamlit_app.py
```
Your browser opens automatically. Fill in the form, click **Predict Risk**, and you'll
get a default-probability score. To stop the app, go back to the terminal and press Ctrl+C.

---

## What if I just want to see the app without training?

The trained models are already included in the `models/` folder, so you can skip step 2
and go straight to step 3 (`streamlit run ui/streamlit_app.py`).

---

## Common problems and fixes

- **"python is not recognized"** → Python isn't installed or not on PATH. Reinstall Python
  and tick "Add Python to PATH". On Mac, use `python3` instead of `python`.
- **"pip is not recognized"** → use `python -m pip install -r requirements.txt` instead.
- **"streamlit is not recognized"** → run `python -m streamlit run ui/streamlit_app.py`.
- **No internet / dataset won't download** → the dataset is already bundled in
  `data/raw/UCI_Credit_Card.csv`, so training works fully offline.
- **Want to retrain differently** → open `src/train.py`, change the line `SCORING = "accuracy"`
  to `SCORING = "f1"` to make the model catch more defaulters (at slightly lower accuracy).

---

## What each file does (for your understanding)

- `data/raw/UCI_Credit_Card.csv` — the dataset (30,000 customers).
- `src/data_loader.py` — loads the data (works offline).
- `src/preprocess.py` — cleans the data and builds the 25 features.
- `src/train.py` — trains 8 models, tunes the best (SGD), saves it.
- `src/evaluate.py` — prints scores and saves charts into `reports/`.
- `ui/streamlit_app.py` — the web app you interact with.
- `models/` — the saved trained model (already included).
- `reports/` — accuracy charts after you run evaluate.
