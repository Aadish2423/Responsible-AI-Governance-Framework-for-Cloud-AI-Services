"""
logs.py — Structured user and audit logging.

Stores logs in‑memory as lists of dicts; provides DataFrame views.
"""

from datetime import datetime
import pandas as pd


class Logger:
    """In‑memory structured logger for the RAI governance framework."""

    def __init__(self):
        self._user_logs: list[dict] = []
        self._audit_logs: list[dict] = []

    # ----- User Logs -----

    def add_user_log(self, user: str, action: str, status: str) -> None:
        self._user_logs.append({
            "user": user,
            "action": action,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": status,
        })

    def get_user_logs(self) -> pd.DataFrame:
        if not self._user_logs:
            return pd.DataFrame(columns=["user", "action", "timestamp", "status"])
        return pd.DataFrame(self._user_logs)

    # ----- Audit Logs -----

    def add_audit_log(
        self,
        input_data: dict,
        prediction: int,
        probability: float,
        bias_score: float,
        decision: str,
        violations: list[str],
    ) -> None:
        self._audit_logs.append({
            "input_data": str(input_data),
            "prediction": prediction,
            "probability": round(probability, 4),
            "bias_score": round(bias_score, 4),
            "decision": decision,
            "policy_violations": "; ".join(violations) if violations else "None",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    def get_audit_logs(self) -> pd.DataFrame:
        if not self._audit_logs:
            return pd.DataFrame(
                columns=[
                    "input_data", "prediction", "probability",
                    "bias_score", "decision", "policy_violations", "timestamp",
                ]
            )
        return pd.DataFrame(self._audit_logs)

    # ----- Compliance report -----

    def generate_compliance_report(self) -> dict:
        """Return a summary dict suitable for a dashboard."""
        audit_df = self.get_audit_logs()
        total = len(audit_df)
        if total == 0:
            return {
                "total_requests": 0,
                "avg_bias_score": 0.0,
                "total_violations": 0,
                "anomaly_count": 0,
            }

        violations_count = int((audit_df["policy_violations"] != "None").sum())
        anomaly_count = int(audit_df["anomaly"].sum()) if "anomaly" in audit_df.columns else 0

        return {
            "total_requests": total,
            "avg_bias_score": round(float(audit_df["bias_score"].mean()), 4),
            "total_violations": violations_count,
            "anomaly_count": anomaly_count,
        }
