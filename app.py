"""
app.py — Streamlit UI for the Responsible AI Governance Framework.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd

from model import generate_data, train_model, governed_predict, FEATURE_COLS
from governance import (
    AccessControl,
    compute_fairness,
    PolicyEngine,
    detect_anomalies,
    BIAS_THRESHOLD,
)
from logs import Logger

# ===================================================================
# Page configuration
# ===================================================================
st.set_page_config(
    page_title="RAI Governance Framework",
    page_icon="🛡️",
    layout="wide",
)

# ===================================================================
# Session‑state initialisation (runs once per session)
# ===================================================================
if "initialized" not in st.session_state:
    # Generate data & train model
    df = generate_data()
    model, scaler = train_model(df)

    st.session_state["df"] = df
    st.session_state["model"] = model
    st.session_state["scaler"] = scaler
    st.session_state["logger"] = Logger()
    st.session_state["initialized"] = True

df: pd.DataFrame = st.session_state["df"]
model = st.session_state["model"]
scaler = st.session_state["scaler"]
logger: Logger = st.session_state["logger"]

# ===================================================================
# Sidebar — role + inputs
# ===================================================================
st.sidebar.title("🛡️ RAI Governance")
st.sidebar.markdown("---")

role = st.sidebar.selectbox("Select Role", AccessControl.get_roles())

st.sidebar.markdown("### Input Features")

income = st.sidebar.number_input("Income ($)", min_value=0, value=60000, step=1000)
credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=850, value=700)
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=35)
debt_ratio = st.sidebar.slider("Debt Ratio", 0.0, 1.0, 0.3, step=0.01)
years_emp = st.sidebar.number_input("Years Employed", min_value=0, value=5)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Non-binary"])

predict_btn = st.sidebar.button("🔮 Predict", use_container_width=True)

# ===================================================================
# Main — Title
# ===================================================================
st.title("🛡️ Responsible AI Governance Framework")
st.caption("Enforcing Fairness · Accountability · Security · Compliance")
st.markdown("---")

# ===================================================================
# Prediction flow
# ===================================================================
if predict_btn:
    # 1) RBAC check
    if not AccessControl.check_access(role, "predict"):
        st.error("❌ **Unauthorized Access** — your role does not allow predictions.")
        logger.add_user_log(user=role, action="predict", status="Unauthorized")
    else:
        # 2) Governed prediction
        input_data = {
            "income": income,
            "credit_score": credit_score,
            "age": age,
            "debt_ratio": debt_ratio,
            "years_emp": years_emp,
        }
        pred_result = governed_predict(model, scaler, input_data)

        # 3) Fairness (integrated into flow)
        fairness = compute_fairness(df, sensitive_attr="gender")

        # 4) Policy engine
        policy = PolicyEngine.evaluate(
            prediction_result=pred_result,
            bias_score=fairness["bias_score"],
            bias_detected=fairness["bias_detected"],
            user_role=role,
            action="predict",
        )

        # 5) Logging
        logger.add_audit_log(
            input_data=input_data,
            prediction=pred_result["prediction"],
            probability=pred_result["probability"],
            bias_score=fairness["bias_score"],
            decision=policy["final_decision"],
            violations=policy["violations"],
        )
        logger.add_user_log(user=role, action="predict", status=policy["final_decision"])

        # 6) Render results ------------------------------------------------
        st.subheader("📊 Prediction Result")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Prediction", "Approved ✅" if pred_result["prediction"] == 1 else "Denied ❌")
        col2.metric("Confidence", pred_result["confidence"])
        col3.metric("Risk Level", pred_result["risk_level"])
        col4.metric("Bias Score", fairness["bias_score"])

        st.markdown(f"**Final Decision:** `{policy['final_decision']}`")

        # 7) Alerts --------------------------------------------------------
        if fairness["bias_detected"]:
            st.warning(
                f"🚨 **Bias Detected** — bias score {fairness['bias_score']} "
                f"exceeds threshold {BIAS_THRESHOLD}"
            )

        if policy["violations"]:
            for v in policy["violations"]:
                if "Unauthorized" in v:
                    st.error(f"❌ {v}")
                else:
                    st.warning(f"⚠️ {v}")

        st.markdown("---")

# ===================================================================
# Logs Section
# ===================================================================
st.subheader("📜 Logs")

if AccessControl.check_access(role, "view_logs"):
    tab_user, tab_audit = st.tabs(["User Logs", "Audit Logs"])

    with tab_user:
        user_df = logger.get_user_logs()
        if user_df.empty:
            st.info("No user logs yet.")
        else:
            st.dataframe(user_df, use_container_width=True)

    with tab_audit:
        audit_df = logger.get_audit_logs()
        if audit_df.empty:
            st.info("No audit logs yet.")
        else:
            st.dataframe(audit_df, use_container_width=True)
else:
    st.error("❌ **Unauthorized** — your role cannot view logs.")

st.markdown("---")

# ===================================================================
# Compliance Dashboard
# ===================================================================
st.subheader("📈 Compliance Dashboard")

if AccessControl.check_access(role, "view_compliance"):
    # --- summary metrics ---
    audit_df = logger.get_audit_logs()
    if not audit_df.empty:
        audit_df = detect_anomalies(audit_df)

    report = logger.generate_compliance_report()
    # Re‑compute anomaly count from detected df
    if not audit_df.empty and "anomaly" in audit_df.columns:
        report["anomaly_count"] = int(audit_df["anomaly"].sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Requests", report["total_requests"])
    m2.metric("Avg Bias Score", report["avg_bias_score"])
    m3.metric("Violations", report["total_violations"])
    m4.metric("Anomalies", report["anomaly_count"])

    # --- fairness metrics ---
    st.markdown("#### Fairness Metrics (Training Data)")
    fairness_full = compute_fairness(df, sensitive_attr="gender")
    st.json(fairness_full)

    # --- anomaly results ---
    if not audit_df.empty and "anomaly" in audit_df.columns:
        anomaly_rows = audit_df[audit_df["anomaly"] == 1]
        st.markdown("#### Anomaly Detection Results")
        if anomaly_rows.empty:
            st.success("No anomalies detected in audit logs.")
        else:
            st.warning(f"🔍 {len(anomaly_rows)} anomalous record(s) found.")
            st.dataframe(anomaly_rows, use_container_width=True)

    # --- violations summary ---
    if not audit_df.empty:
        violations_df = audit_df[audit_df["policy_violations"] != "None"]
        st.markdown("#### Violations Summary")
        if violations_df.empty:
            st.success("No policy violations recorded.")
        else:
            st.dataframe(violations_df[["timestamp", "decision", "policy_violations"]], use_container_width=True)
else:
    st.error("❌ **Unauthorized** — your role cannot view the compliance dashboard.")
