# Credit Card Default Prediction

Predicts whether a credit card client will default next month, using the UCI
"Default of Credit Card Clients" (Taiwan) dataset. Eight models trained and
compared; a tuned SGD classifier is selected for its precision–recall balance.

## Quickstart
```bash
pip install -r requirements.txt
python src/train.py        # downloads data, trains 8 models, tunes SGD, saves artifacts
python src/evaluate.py     # metrics + confusion-matrix & ROC plots -> reports/
streamlit run ui/streamlit_app.py   # launch the risk-scoring UI
```

## Pipeline
clean & rename -> engineer to 25 features -> stratified split + StandardScaler
-> class_weight balanced (22% imbalance) -> train 8 models -> GridSearch the SGD
-> evaluate on accuracy / precision / recall / F1 / ROC-AUC.

## Structure
- `src/preprocess.py` — cleaning + feature engineering (23 raw + 2 engineered = 25)
- `src/train.py` — trains all 8 models, tunes SGD, saves model/scaler/defaults
- `src/evaluate.py` — classification report + plots
- `ui/streamlit_app.py` — interactive predictor (collects 9 key inputs, auto-fills rest)
- `models/` — saved `sgd_best.joblib`, `scaler.joblib`, `defaults.joblib`
