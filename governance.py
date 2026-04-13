"""
governance.py — Policy engine, fairness metrics, RBAC, and anomaly detection.

Centralises all governance logic for the RAI framework.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
)

# ===================================================================
# Access Control (RBAC)
# ===================================================================

# Allowed actions per role
_ROLE_PERMISSIONS: dict[str, list[str]] = {
    "admin": ["predict", "view_logs", "view_fairness", "view_compliance", "full"],
    "auditor": ["view_logs", "view_fairness", "view_compliance"],
    "user": ["predict"],
}


class AccessControl:
    """Simple role‑based access control."""

    @staticmethod
    def check_access(role: str, action: str) -> bool:
        perms = _ROLE_PERMISSIONS.get(role, [])
        return action in perms or "full" in perms

    @staticmethod
    def get_roles() -> list[str]:
        return list(_ROLE_PERMISSIONS.keys())


# ===================================================================
# Fairness metrics
# ===================================================================

BIAS_THRESHOLD = 0.1


def compute_fairness(
    df: pd.DataFrame,
    sensitive_attr: str = "gender",
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
) -> dict:
    """
    Compute fairness metrics integrated into the prediction flow.

    Returns
    -------
    dict with:
        demographic_parity_diff – float
        equalized_odds_diff     – float
        bias_score              – float  (max of the two)
        bias_detected           – bool
    """
    y_true = df[y_true_col]
    y_pred = df[y_pred_col]
    sensitive = df[sensitive_attr]

    dp_diff = abs(demographic_parity_difference(
        y_true=y_true, y_pred=y_pred, sensitive_features=sensitive
    ))
    eo_diff = abs(equalized_odds_difference(
        y_true=y_true, y_pred=y_pred, sensitive_features=sensitive
    ))

    bias_score = round(max(dp_diff, eo_diff), 4)

    return {
        "demographic_parity_diff": round(dp_diff, 4),
        "equalized_odds_diff": round(eo_diff, 4),
        "bias_score": bias_score,
        "bias_detected": bias_score > BIAS_THRESHOLD,
    }


# ===================================================================
# Policy Engine
# ===================================================================

class PolicyEngine:
    """
    Enforce governance policies on each prediction request.

    Checks:
      1. Bias threshold  (0.1)
      2. Confidence threshold  (via prediction result)
      3. RBAC permission
    """

    CONFIDENCE_THRESHOLD_LABEL = "Low"  # matches model.py output

    @staticmethod
    def evaluate(
        prediction_result: dict,
        bias_score: float,
        bias_detected: bool,
        user_role: str,
        action: str = "predict",
    ) -> dict:
        """
        Returns
        -------
        dict with:
            final_decision – str ("Approved" / "Rejected" /
                                  "Review Required" / "Policy Violation")
            violations     – list[str]
        """
        violations: list[str] = []

        # --- RBAC check ---
        if not AccessControl.check_access(user_role, action):
            violations.append(f"Unauthorized: role '{user_role}' cannot perform '{action}'")

        # --- Bias check ---
        if bias_detected:
            violations.append(
                f"Bias threshold exceeded: bias_score={bias_score} > {BIAS_THRESHOLD}"
            )

        # --- Confidence check ---
        if prediction_result.get("confidence") == PolicyEngine.CONFIDENCE_THRESHOLD_LABEL:
            violations.append("Low confidence — manual review required")

        # Determine final decision
        if any("Unauthorized" in v for v in violations):
            final_decision = "Policy Violation"
        elif any("Bias" in v for v in violations):
            final_decision = "Rejected"
        elif any("confidence" in v.lower() for v in violations):
            final_decision = "Review Required"
        else:
            final_decision = prediction_result.get("decision", "Approved")

        return {
            "final_decision": final_decision,
            "violations": violations,
        }


# ===================================================================
# Anomaly Detection
# ===================================================================

def detect_anomalies(audit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Isolation Forest on audit logs to flag anomalous predictions.

    Expects columns: ``probability``, ``bias_score``.
    Adds an ``anomaly`` column (1 = anomaly, 0 = normal).
    """
    if audit_df.empty:
        return audit_df.assign(anomaly=pd.Series(dtype=int))

    numeric_cols = ["probability", "bias_score"]
    available = [c for c in numeric_cols if c in audit_df.columns]
    if not available:
        return audit_df.assign(anomaly=0)

    X = audit_df[available].values
    iso = IsolationForest(contamination=0.1, random_state=42)
    preds = iso.fit_predict(X)  # 1 = normal, -1 = anomaly
    audit_df = audit_df.copy()
    audit_df["anomaly"] = (preds == -1).astype(int)
    return audit_df
