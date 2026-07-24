"""Train 8 models, compare, tune the SGD, and save model + scaler + defaults."""
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from preprocess import build_features, FEATURE_ORDER
from data_loader import load_raw

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")

# How to pick the "best" tuned SGD:
#   "accuracy" -> ~80% accuracy, more conservative (fewer false alarms)
#   "f1"       -> ~77% accuracy, catches more defaulters (better recall/balance)
SCORING = "accuracy"


def load_data():
    return build_features(load_raw())


def get_models():
    return {
        "SGD":        SGDClassifier(loss="log_loss", class_weight="balanced", random_state=42),
        "LogReg":     LogisticRegression(max_iter=1000, class_weight="balanced"),
        "DecTree":    DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "RandForest": RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=42),
        "GradBoost":  GradientBoostingClassifier(random_state=42),
        "SVM":        SVC(probability=True, class_weight="balanced"),
        "KNN":        KNeighborsClassifier(),
        "NaiveBayes": GaussianNB(),
    }


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    df = load_data()
    X, y = df[FEATURE_ORDER], df["DEFAULT"]
    print(f"Data: {X.shape}, default rate: {y.mean():.1%}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    # ---- Train & compare all 8 ----
    rows = []
    for name, m in get_models().items():
        m.fit(X_tr_s, y_tr)
        pred = m.predict(X_te_s)
        proba = m.predict_proba(X_te_s)[:, 1] if hasattr(m, "predict_proba") else pred
        rows.append({"model": name,
                     "accuracy": accuracy_score(y_te, pred),
                     "f1": f1_score(y_te, pred),
                     "roc_auc": roc_auc_score(y_te, proba)})
    comp = pd.DataFrame(rows).sort_values("f1", ascending=False)
    comp.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"), index=False)
    print("\n", comp.to_string(index=False))

    # ---- Tune SGD (let the search decide how much to balance) ----
    grid = {"loss": ["log_loss", "modified_huber"],
            "alpha": [1e-4, 1e-3, 1e-2],
            "penalty": ["l2", "elasticnet"],
            "l1_ratio": [0.15, 0.5],
            "class_weight": [None, "balanced", {0: 1, 1: 2}]}
    gs = GridSearchCV(SGDClassifier(random_state=42),
                      grid, scoring=SCORING, cv=5, n_jobs=-1)
    gs.fit(X_tr_s, y_tr)
    best = gs.best_estimator_
    acc = accuracy_score(y_te, best.predict(X_te_s))
    f1 = f1_score(y_te, best.predict(X_te_s))
    auc = roc_auc_score(y_te, best.predict_proba(X_te_s)[:, 1])
    print(f"\nBest SGD params: {gs.best_params_}")
    print(f"Tuned SGD -> accuracy: {acc:.1%} | F1: {f1:.3f} | ROC-AUC: {auc:.3f}")

    # ---- Save everything the UI needs ----
    joblib.dump(best, os.path.join(MODELS_DIR, "sgd_best.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    # Median of every feature -> used to auto-fill inputs the UI doesn't collect
    joblib.dump(X.median().to_dict(), os.path.join(MODELS_DIR, "defaults.joblib"))
    print("\nSaved: sgd_best.joblib, scaler.joblib, defaults.joblib")


if __name__ == "__main__":
    main()
