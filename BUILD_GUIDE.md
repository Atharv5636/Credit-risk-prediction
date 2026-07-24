# Credit Card Default Prediction — Complete Build Guide (Start → End)

Build target: **UCI "Default of Credit Card Clients" (Taiwan) dataset → 8 ML models → tuned SGD classifier (~83% accuracy) → web UI for risk scoring.** Designed to be built inside **Google Antigravity** by dispatching agents per stage.

---

## 0. The 30-second mental model

```
Raw CSV (30,000 clients, 23 features, 22% default)
        │
   [1] Load + sanity check
        │
   [2] Clean + rename + fix categorical noise
        │
   [3] EDA (understand defaults vs non-defaults)
        │
   [4] Feature engineering → 25 features
        │
   [5] Train/test split (stratified) + scaling
        │
   [6] Handle imbalance (class_weight / SMOTE)
        │
   [7] Train 8 models → compare
        │
   [8] Hyperparameter-tune the front-runners (SGD wins)
        │
   [9] Evaluate: precision, recall, F1, ROC-AUC
        │
  [10] Save model (joblib) → serve via FastAPI/Streamlit UI
        │
  [11] Deploy
```

---

## 1. The dataset

**Source:** UCI Machine Learning Repository — "Default of Credit Card Clients Dataset" (also mirrored on Kaggle). It's a single CSV, ~30,000 rows, no real download friction.

- Rows: **30,000** credit card holders in Taiwan (Apr–Sep 2005)
- Target `Y` = **default.payment.next.month** (1 = will default, 0 = won't)
- Class balance: **~22% defaulters** (6,636 of 30,000) — imbalanced, this is why you weight/resample
- 23 raw explanatory features:

| Feature | Meaning |
|---|---|
| `LIMIT_BAL` | Credit limit (NT dollars) |
| `SEX` | 1=male, 2=female |
| `EDUCATION` | 1=grad school, 2=university, 3=high school, 4=others (5,6,0 = unknown → clean these) |
| `MARRIAGE` | 1=married, 2=single, 3=others (0 = unknown → clean) |
| `AGE` | years |
| `PAY_0, PAY_2 … PAY_6` | Repayment status months Sep→Apr (-1=paid duly, 1=1mo late … 8=8mo late). **Strongest predictors.** |
| `BILL_AMT1 … BILL_AMT6` | Bill statement amount, Sep→Apr |
| `PAY_AMT1 … PAY_AMT6` | Previous payment amount, Sep→Apr |

> Note: column is named `PAY_0` in the raw file (not `PAY_1`) — a classic gotcha. Rename it.

**Getting it (any one):**
```bash
# Option A – ucimlrepo package (cleanest)
pip install ucimlrepo
```
```python
from ucimlrepo import fetch_ucirepo
ds = fetch_ucirepo(id=350)            # 350 = default of credit card clients
X = ds.data.features
y = ds.data.targets
```
```bash
# Option B – Kaggle
kaggle datasets download -d uciml/default-of-credit-card-clients-dataset
```

---

## 2. Tech stack & project structure

**Stack:** Python 3.11 · pandas · numpy · scikit-learn · imbalanced-learn · matplotlib/seaborn · joblib · FastAPI (API) · Streamlit (quick UI) **or** React (polished UI).

```
credit-default-predictor/
├── data/
│   ├── raw/credit.csv
│   └── processed/clean.parquet
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── config.py            # paths, constants, column names
│   ├── data_loader.py       # load + initial validation
│   ├── preprocess.py        # clean, rename, engineer, scale
│   ├── train.py             # train all 8 models + tune
│   ├── evaluate.py          # metrics + plots
│   └── predict.py           # load model, single/batch inference
├── models/
│   ├── sgd_best.joblib      # final model
│   └── scaler.joblib        # fitted StandardScaler
├── api/
│   └── main.py              # FastAPI app
├── ui/
│   ├── streamlit_app.py     # OR
│   └── react-frontend/      # Vite + React + Tailwind
├── reports/
│   └── model_comparison.csv
├── requirements.txt
├── Dockerfile
└── README.md
```

`requirements.txt`:
```
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
joblib
fastapi
uvicorn[standard]
streamlit
ucimlrepo
pydantic
```

---

## 3. Stage-by-stage pipeline (with real code)

### [1] Load + sanity check — `data_loader.py`
```python
import pandas as pd
from ucimlrepo import fetch_ucirepo

def load_raw():
    ds = fetch_ucirepo(id=350)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    return df

if __name__ == "__main__":
    df = load_raw()
    print(df.shape)              # (30000, 24)
    print(df.isna().sum().sum()) # expect 0
    print(df.iloc[:, -1].value_counts(normalize=True))  # ~0.78 / 0.22
```

### [2] Clean + rename — `preprocess.py`
```python
import pandas as pd

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.rename(columns={"PAY_0": "PAY_1",
                            "default payment next month": "DEFAULT",
                            "default.payment.next.month": "DEFAULT"})
    # EDUCATION: collapse unknowns (0,5,6) into "others" (4)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    # MARRIAGE: collapse unknown (0) into "others" (3)
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    return df
```

### [3] EDA (do this in the notebook)
Look at: default rate by `PAY_1` (steep — late payers default far more), by `LIMIT_BAL` buckets, by `EDUCATION`/`AGE`. Correlation heatmap of `BILL_AMT*` (they're highly collinear → candidates for engineering). This is what justifies your feature choices.

### [4] Feature engineering → **25 features**
Start from 23 raw, drop nothing essential, and **add engineered features** to reach ~25 and boost signal:
```python
def engineer(df):
    df = df.copy()
    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    pay_cols  = [f"PAY_AMT{i}"  for i in range(1, 7)]
    delay_cols = [f"PAY_{i}"    for i in range(1, 7)]

    # +1: average utilization (bill / limit) — strong risk signal
    df["AVG_UTILIZATION"] = df[bill_cols].mean(axis=1) / df["LIMIT_BAL"].replace(0, 1)
    # +1: total months delayed (sum of positive delays) — captures chronic lateness
    df["TOTAL_DELAY"] = df[delay_cols].clip(lower=0).sum(axis=1)
    # +1: payment-to-bill ratio (how much of the bill they actually pay)
    df["PAY_TO_BILL"] = df[pay_cols].sum(axis=1) / (df[bill_cols].sum(axis=1).abs() + 1)
    return df
# 23 raw + 3 engineered = 26 → drop the redundant raw PAY_0 dup or one BILL col to land at 25
```
> The exact count (24/25/26) is flexible — the point on a resume is "engineered features beyond the raw set." Keep whichever improve validation F1.

### [5] Split + scale
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

X = df.drop(columns=["DEFAULT"]); y = df["DEFAULT"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler().fit(X_tr)
X_tr_s = scaler.transform(X_tr)
X_te_s = scaler.transform(X_te)
joblib.dump(scaler, "models/scaler.joblib")
```
> Scaling matters a lot for **SGD, Logistic Regression, SVM, KNN**. Tree models don't need it, but scaling them does no harm.

### [6] Handle the 22% imbalance
Two valid routes — pick one and stay consistent:
- **`class_weight="balanced"`** (cheap, no synthetic data) — works for SGD, LogReg, SVM, RandomForest.
- **SMOTE** (oversample minority) — fit **only on training data**:
```python
from imblearn.over_sampling import SMOTE
X_tr_s, y_tr = SMOTE(random_state=42).fit_resample(X_tr_s, y_tr)
```

### [7] Train all 8 models — `train.py`
```python
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, roc_auc_score, classification_report

models = {
    "SGD":        SGDClassifier(loss="log_loss", class_weight="balanced", random_state=42),
    "LogReg":     LogisticRegression(max_iter=1000, class_weight="balanced"),
    "DecTree":    DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "RandForest": RandomForestClassifier(class_weight="balanced", random_state=42),
    "GradBoost":  GradientBoostingClassifier(random_state=42),
    "SVM":        SVC(probability=True, class_weight="balanced"),
    "KNN":        KNeighborsClassifier(),
    "NaiveBayes": GaussianNB(),
}

results = []
for name, m in models.items():
    m.fit(X_tr_s, y_tr)
    pred = m.predict(X_te_s)
    proba = m.predict_proba(X_te_s)[:, 1] if hasattr(m, "predict_proba") else pred
    results.append({"model": name,
                    "f1": f1_score(y_te, pred),
                    "roc_auc": roc_auc_score(y_te, proba)})
import pandas as pd
print(pd.DataFrame(results).sort_values("f1", ascending=False))
```

### [8] Tune the front-runners (SGD wins on precision–recall balance)
```python
from sklearn.model_selection import GridSearchCV
grid = {
    "loss": ["log_loss", "modified_huber"],
    "alpha": [1e-4, 1e-3, 1e-2],
    "penalty": ["l2", "elasticnet"],
    "l1_ratio": [0.15, 0.5],
}
gs = GridSearchCV(SGDClassifier(class_weight="balanced", random_state=42),
                  grid, scoring="f1", cv=5, n_jobs=-1)
gs.fit(X_tr_s, y_tr)
best = gs.best_estimator_
joblib.dump(best, "models/sgd_best.joblib")
print(gs.best_params_)
```

### [9] Evaluate — `evaluate.py`
Report **all four** metrics (accuracy alone lies on imbalanced data):
- **Accuracy** → your headline ~83%
- **Precision** (of predicted defaulters, how many truly default) → controls false alarms on good customers
- **Recall** (of true defaulters, how many you caught) → controls missed risk
- **F1** → balance of the two (your selection criterion)
- **ROC-AUC** → ranking quality across thresholds

```python
from sklearn.metrics import (classification_report, confusion_matrix,
                             RocCurveDisplay, ConfusionMatrixDisplay)
print(classification_report(y_te, best.predict(X_te_s)))
RocCurveDisplay.from_estimator(best, X_te_s, y_te)
```
> **Why SGD as the pick (your resume line):** strong precision–recall balance + fast/lightweight + linear & interpretable (coefficients = feature weights), so it reduces false positives on low-risk customers without the cost of a heavy ensemble. Defensible in an interview.

### [10] Inference — `predict.py`
```python
import joblib, numpy as np
model = joblib.load("models/sgd_best.joblib")
scaler = joblib.load("models/scaler.joblib")

def predict_one(features: dict):
    import pandas as pd
    X = pd.DataFrame([features])
    Xs = scaler.transform(X)
    proba = float(model.predict_proba(Xs)[0, 1])
    return {"default_probability": round(proba, 3),
            "decision": "HIGH RISK" if proba >= 0.5 else "LOW RISK"}
```

---

## 4. The UI

Two paths — pick based on how polished you want it.

### Path A — Streamlit (fastest, one file)
`ui/streamlit_app.py`:
```python
import streamlit as st, joblib, pandas as pd

model = joblib.load("models/sgd_best.joblib")
scaler = joblib.load("models/scaler.joblib")

st.title("Credit Card Default Risk Predictor")
st.caption("Tuned SGD classifier · UCI Taiwan dataset")

col1, col2 = st.columns(2)
with col1:
    limit = st.number_input("Credit Limit (NT$)", 10000, 1000000, 200000, step=10000)
    age   = st.slider("Age", 21, 79, 35)
    pay_1 = st.selectbox("Last month repayment status",
                         options=[-1,0,1,2,3,4,5,6,7,8],
                         format_func=lambda x: "Paid duly" if x<=0 else f"{x} mo late")
with col2:
    bill  = st.number_input("Latest bill amount (NT$)", 0, 1000000, 50000, step=5000)
    paid  = st.number_input("Latest payment (NT$)", 0, 1000000, 5000, step=1000)
    edu   = st.selectbox("Education", [1,2,3,4],
                         format_func=lambda x: {1:"Grad",2:"University",3:"High school",4:"Other"}[x])

if st.button("Predict Risk", type="primary"):
    row = {  # fill all 25 features; use sensible defaults for the rest
        "LIMIT_BAL": limit, "AGE": age, "PAY_1": pay_1,
        "BILL_AMT1": bill, "PAY_AMT1": paid, "EDUCATION": edu,
        # ... map remaining features ...
    }
    Xs = scaler.transform(pd.DataFrame([row]))
    p = float(model.predict_proba(Xs)[0,1])
    st.metric("Default Probability", f"{p:.1%}")
    st.progress(min(p,1.0))
    (st.error if p>=0.5 else st.success)("HIGH RISK" if p>=0.5 else "LOW RISK")
```
Run: `streamlit run ui/streamlit_app.py`

### Path B — FastAPI backend + React frontend (portfolio-grade)
`api/main.py`:
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, pandas as pd

app = FastAPI(title="Credit Default API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
model = joblib.load("models/sgd_best.joblib")
scaler = joblib.load("models/scaler.joblib")

class Applicant(BaseModel):
    LIMIT_BAL: float; AGE: int; PAY_1: int
    BILL_AMT1: float; PAY_AMT1: float; EDUCATION: int
    # ... rest of the 25 fields

@app.post("/predict")
def predict(a: Applicant):
    Xs = scaler.transform(pd.DataFrame([a.dict()]))
    p = float(model.predict_proba(Xs)[0,1])
    return {"probability": round(p,3), "risk": "HIGH" if p>=0.5 else "LOW"}
```
Run: `uvicorn api.main:app --reload`. The React app `fetch`es `POST /predict` and renders a gauge/result card. (Antigravity can scaffold the whole React frontend from a one-line prompt — see below.)

---

## 5. Deployment
- **Streamlit Community Cloud** — free, push repo, point to `streamlit_app.py`. Easiest.
- **Render / Railway / Fly.io** — for the FastAPI + React combo. Add the `Dockerfile`.
- **Docker** sketch:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt . && RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 6. How to build this **in Google Antigravity**

Antigravity is agent-first: you describe a task, an agent **plans → writes code → runs it → verifies** and hands back artifacts (plan, diffs, screenshots). Best approach is to dispatch **one agent per stage** rather than asking for the whole app at once. Use **Plan mode** for the big stages so you can review the plan before code is written.

Recommended prompts to feed the agent, in order:

1. *"Create a Python project with this structure [paste the tree from section 2]. Add requirements.txt and a README."*
2. *"In `src/data_loader.py`, load the UCI 'default of credit card clients' dataset (id 350) via ucimlrepo, concat features+target, and print shape and class balance."*
3. *"In `src/preprocess.py`, implement clean(), engineer() and a build_features() pipeline exactly per these specs [paste section 3 steps 2 & 4]. Reach 25 features."*
4. *"In `src/train.py`, train these 8 sklearn models with class_weight balanced, do a stratified 80/20 split with StandardScaler, and output a sorted comparison table of F1 and ROC-AUC."*
5. *"Add a GridSearchCV tuning block for the SGDClassifier optimizing F1, save best model and scaler to /models with joblib."*
6. *"In `src/evaluate.py`, print classification_report and save confusion-matrix + ROC-curve plots to /reports."*
7. *"Build a FastAPI app in `api/main.py` with a /predict endpoint, then scaffold a Vite + React + Tailwind frontend in `ui/react-frontend` with a form for the key inputs and a risk gauge that calls /predict."*

Because Antigravity can drive a browser, ask it to **launch the app and screenshot the prediction working** — that screenshot artifact is your proof the UI works end to end. Leave Google-Docs-style comments on its implementation plan to redirect before it writes code.

---

## 7. Mapping back to your résumé bullets
| Résumé claim | Where it's earned |
|---|---|
| 8 ML models trained & tuned, 83% acc, tuned SGD best | Sections 3.7–3.8 |
| 25-feature dataset, cleaning/renaming/scaling, 22% imbalance | Sections 3.2, 3.4–3.6 |
| Evaluated via precision/recall/F1/ROC-AUC | Section 3.9 |
| Selected SGD for precision–recall balance, fewer false positives | Section 3.9 rationale |

---

### Suggested build order for a weekend
**Day 1:** sections 1–3 (data → models → tuning → saved model). **Day 2:** sections 4–6 (UI → deploy) in Antigravity, capture screenshots, write README.
