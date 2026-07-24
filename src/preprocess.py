"""Data cleaning + feature engineering. Final feature count = 25."""
import pandas as pd

RAW_FEATURES = (
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + [f"PAY_{i}" for i in range(1, 7)]       # PAY_1..PAY_6 (repayment status)
    + [f"BILL_AMT{i}" for i in range(1, 7)]   # bill amounts
    + [f"PAY_AMT{i}" for i in range(1, 7)]    # payment amounts
)  # 23 raw features

ENGINEERED = ["AVG_UTILIZATION", "TOTAL_DELAY"]  # +2 -> 25 total
FEATURE_ORDER = RAW_FEATURES + ENGINEERED


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.rename(columns={
        "PAY_0": "PAY_1",
        "default payment next month": "DEFAULT",
        "default.payment.next.month": "DEFAULT",
    })
    # Collapse unknown/garbage categories into "others"
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    delay_cols = [f"PAY_{i}" for i in range(1, 7)]
    # Average credit utilization: mean bill / credit limit (clipped at sensible range)
    df["AVG_UTILIZATION"] = (df[bill_cols].mean(axis=1)
                             / df["LIMIT_BAL"].replace(0, 1)).clip(0, 5)
    # Chronic lateness: total months delayed across the 6-month window
    df["TOTAL_DELAY"] = df[delay_cols].clip(lower=0).sum(axis=1)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: clean -> engineer -> select ordered 25 features (+ DEFAULT if present)."""
    df = engineer(clean(df))
    cols = FEATURE_ORDER + (["DEFAULT"] if "DEFAULT" in df.columns else [])
    return df[cols]
