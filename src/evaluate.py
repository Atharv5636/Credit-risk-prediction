"""Evaluate the saved model: full metrics + confusion matrix + ROC curve."""
import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, ConfusionMatrixDisplay,
                             RocCurveDisplay)

from preprocess import FEATURE_ORDER
from train import load_data, MODELS_DIR, REPORTS_DIR


def main():
    model = joblib.load(os.path.join(MODELS_DIR, "sgd_best.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))

    df = load_data()
    X, y = df[FEATURE_ORDER], df["DEFAULT"]
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_te_s = scaler.transform(X_te)

    print(classification_report(y_te, model.predict(X_te_s),
                                target_names=["No Default", "Default"]))

    ConfusionMatrixDisplay.from_predictions(y_te, model.predict(X_te_s))
    plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"), bbox_inches="tight")
    plt.close()

    RocCurveDisplay.from_estimator(model, X_te_s, y_te)
    plt.savefig(os.path.join(REPORTS_DIR, "roc_curve.png"), bbox_inches="tight")
    plt.close()
    print("Saved plots to reports/")


if __name__ == "__main__":
    main()
