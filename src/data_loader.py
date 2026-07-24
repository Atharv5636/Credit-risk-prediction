"""Robust data loading. Tries, in order:
  1. Local CSV bundled in data/raw/  (works fully offline — recommended)
  2. ucimlrepo package download
  3. GitHub mirror download
Whichever succeeds first is used. This avoids the common UCI SSL / network errors.
"""
import os
import ssl
import urllib.request
import pandas as pd

HERE = os.path.dirname(__file__)
LOCAL_CSV = os.path.join(HERE, "..", "data", "raw", "UCI_Credit_Card.csv")
MIRROR_URL = ("https://raw.githubusercontent.com/YuChenAmberLu/"
              "Data-Science--Credit-Card-Default/master/UCI_Credit_Card.csv")


def _from_local():
    if os.path.exists(LOCAL_CSV):
        print(f"[data] Loading bundled CSV: {LOCAL_CSV}")
        return pd.read_csv(LOCAL_CSV)
    return None


def _from_ucimlrepo():
    try:
        from ucimlrepo import fetch_ucirepo
        print("[data] Downloading via ucimlrepo (id=350)...")
        ds = fetch_ucirepo(id=350)
        return pd.concat([ds.data.features, ds.data.targets], axis=1)
    except Exception as e:
        print(f"[data] ucimlrepo failed ({e}); trying mirror...")
        return None


def _from_mirror():
    try:
        print("[data] Downloading from GitHub mirror...")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE  # tolerate strict-SSL environments
        os.makedirs(os.path.dirname(LOCAL_CSV), exist_ok=True)
        with urllib.request.urlopen(MIRROR_URL, context=ctx, timeout=30) as r:
            with open(LOCAL_CSV, "wb") as f:
                f.write(r.read())
        return pd.read_csv(LOCAL_CSV)
    except Exception as e:
        print(f"[data] Mirror failed ({e}).")
        return None


def load_raw() -> pd.DataFrame:
    for loader in (_from_local, _from_ucimlrepo, _from_mirror):
        df = loader()
        if df is not None and len(df) > 1000:
            print(f"[data] Loaded {df.shape[0]} rows, {df.shape[1]} columns.")
            return df
    raise RuntimeError(
        "Could not load the dataset by any method. Download UCI_Credit_Card.csv "
        "from Kaggle and place it in data/raw/."
    )
