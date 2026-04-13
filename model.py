"""
model.py — Model training and governed prediction.

Encapsulates:
  - Synthetic data generation (loan‑approval dataset with intentional bias).
  - Logistic Regression training.
  - Governed prediction with confidence & risk classification.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_COLS = ["income", "credit_score", "age", "debt_ratio", "years_emp"]
CONFIDENCE_THRESHOLD = 0.6

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Produce a synthetic loan‑approval dataset with protected attributes
    and intentional demographic bias (mirrors the notebook's
    ``generate_fairness_benchmark``).
    """
    rng = np.random.RandomState(seed)

    gender = rng.choice(["Male", "Female", "Non-binary"], n, p=[0.50, 0.45, 0.05])
    credit_score = rng.normal(680, 80, n).clip(300, 850).astype(int)
    income = rng.lognormal(10.8, 0.6, n).astype(int)
    age = rng.randint(18, 70, n)
    debt_ratio = rng.beta(2, 5, n).round(3)
    years_emp = rng.poisson(7, n)

    # Ground truth based on objective features
    score = (
        (credit_score - 300) / 550 * 0.5
        + (income / 200_000) * 0.3
        - debt_ratio * 0.2
    )
    y_true = (score + rng.normal(0, 0.08, n) > 0.38).astype(int)

    # Inject demographic bias into predictions
    bias_term = np.zeros(n)
    bias_term[gender == "Female"] -= 0.07
    bias_term[gender == "Non-binary"] -= 0.12

    pred_prob = (score + bias_term + rng.normal(0, 0.06, n)).clip(0, 1)
    y_pred = (pred_prob > 0.40).astype(int)

    return pd.DataFrame({
        "gender": gender,
        "credit_score": credit_score,
        "income": income,
        "age": age,
        "debt_ratio": debt_ratio,
        "years_emp": years_emp,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_prob": pred_prob.round(4),
    })


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(df: pd.DataFrame):
    """
    Train a simple Logistic Regression on the objective features.

    Returns
    -------
    model : LogisticRegression
    scaler : StandardScaler
    """
    X = df[FEATURE_COLS].values
    y = df["y_true"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler


# ---------------------------------------------------------------------------
# Risk classification helper
# ---------------------------------------------------------------------------

def _classify_risk(probability: float) -> str:
    if probability > 0.8:
        return "Low Risk"
    elif probability >= 0.5:
        return "Medium Risk"
    else:
        return "High Risk"


# ---------------------------------------------------------------------------
# Governed prediction
# ---------------------------------------------------------------------------

def governed_predict(model, scaler, input_dict: dict) -> dict:
    """
    Perform a *governed* prediction.

    Parameters
    ----------
    model : fitted LogisticRegression
    scaler : fitted StandardScaler
    input_dict : dict with keys matching ``FEATURE_COLS``

    Returns
    -------
    dict with:
        prediction   – 0 or 1
        probability  – float  (probability of positive class)
        confidence   – str    ("High" or "Low")
        decision     – str    ("Approved" / "Denied" / "Review Required")
        risk_level   – str    ("High Risk" / "Medium Risk" / "Low Risk")
    """
    features = np.array([[input_dict[c] for c in FEATURE_COLS]])
    features_scaled = scaler.transform(features)

    proba = model.predict_proba(features_scaled)[0]  # [P(0), P(1)]
    prob_positive = float(proba[1])
    prediction = int(prob_positive >= 0.5)

    # Confidence gate
    confidence = max(proba)
    if confidence < CONFIDENCE_THRESHOLD:
        decision = "Review Required"
        confidence_label = "Low"
    else:
        decision = "Approved" if prediction == 1 else "Denied"
        confidence_label = "High"

    risk_level = _classify_risk(prob_positive)

    return {
        "prediction": prediction,
        "probability": round(prob_positive, 4),
        "confidence": confidence_label,
        "decision": decision,
        "risk_level": risk_level,
    }
