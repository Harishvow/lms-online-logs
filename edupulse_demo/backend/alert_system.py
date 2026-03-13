"""
EduPulse – Alert System Module
Generates and manages teacher alerts for at-risk students.
"""

import pandas as pd
from datetime import datetime


# ─── Configuration ───────────────────────────────────────────────────────────
ENGAGEMENT_THRESHOLD = 30.0  # Below this score triggers an alert
TREND_THRESHOLD = -1.5       # Engagement trend below this triggers an alert


def generate_alerts(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate alerts for students who meet risk criteria.
    
    Alert triggers:
        1. engagement_score < ENGAGEMENT_THRESHOLD
        2. risk_label == "High Risk"
        3. engagement_trend < TREND_THRESHOLD
    
    Returns:
        DataFrame of alerts with student details and alert messages.
    """
    alerts = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for _, student in summary_df.iterrows():
        alert_reasons = []
        severity = "info"

        # Check engagement score
        if student.get("avg_engagement_score", 100) < ENGAGEMENT_THRESHOLD:
            alert_reasons.append("Low Engagement Score")
            severity = "critical"

        # Check risk label
        if student.get("risk_label") == "High Risk":
            alert_reasons.append("High Risk Classification")
            severity = "critical"
        elif student.get("risk_label") == "Moderate Risk":
            if not alert_reasons:
                alert_reasons.append("Moderate Risk Classification")
                severity = "warning"

        # Check engagement trend
        if student.get("engagement_trend", 0) < TREND_THRESHOLD:
            alert_reasons.append("Drop-off Behavior Detected")
            severity = "critical"

        # Check cluster
        if student.get("cluster_label") == "At Risk / Drop-off":
            if "Low Engagement Score" not in alert_reasons:
                alert_reasons.append("At-Risk Cluster Assignment")
            if severity != "critical":
                severity = "warning"

        if alert_reasons:
            message = _format_alert_message(student, alert_reasons)
            alerts.append({
                "student_id": int(student["student_id"]),
                "engagement_score": round(student.get("avg_engagement_score", 0), 1),
                "risk_level": student.get("risk_label", "Unknown"),
                "cluster": student.get("cluster_label", "Unknown"),
                "engagement_trend": round(student.get("engagement_trend", 0), 2),
                "severity": severity,
                "alert_reasons": " | ".join(alert_reasons),
                "alert_message": message,
                "timestamp": timestamp,
                "status": "Active",
            })

    alerts_df = pd.DataFrame(alerts)

    if not alerts_df.empty:
        # Sort by severity (critical first) then by engagement score (lowest first)
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        alerts_df["severity_rank"] = alerts_df["severity"].map(severity_order)
        alerts_df = alerts_df.sort_values(
            ["severity_rank", "engagement_score"]
        ).reset_index(drop=True)
        alerts_df = alerts_df.drop(columns=["severity_rank"])

    print(f"\n🚨 Alert System: Generated {len(alerts_df)} alerts")
    if not alerts_df.empty:
        critical = (alerts_df["severity"] == "critical").sum()
        warning = (alerts_df["severity"] == "warning").sum()
        print(f"   🔴 Critical: {critical}")
        print(f"   🟡 Warning: {warning}")

    return alerts_df


def _format_alert_message(student: pd.Series, reasons: list) -> str:
    """Format a human-readable alert message for a student."""
    sid = int(student["student_id"])
    score = student.get("avg_engagement_score", 0)
    trend = student.get("engagement_trend", 0)

    msg = f"⚠️ Early Warning: Student {sid} engagement is critically low. "
    msg += f"Score: {score:.1f}/100, Trend: {trend:+.2f}. "
    msg += f"Reasons: {', '.join(reasons)}. "
    msg += "Teacher intervention recommended."
    return msg


def get_alert_summary(alerts_df: pd.DataFrame) -> dict:
    """Get a summary of the current alert state."""
    if alerts_df.empty:
        return {
            "total_alerts": 0,
            "critical": 0,
            "warning": 0,
            "top_alerts": [],
        }

    return {
        "total_alerts": len(alerts_df),
        "critical": int((alerts_df["severity"] == "critical").sum()),
        "warning": int((alerts_df["severity"] == "warning").sum()),
        "top_alerts": alerts_df.head(5).to_dict("records"),
    }


def format_alerts_for_display(alerts_df: pd.DataFrame) -> list:
    """Format alerts for Streamlit display."""
    display_alerts = []
    for _, row in alerts_df.iterrows():
        icon = "🔴" if row["severity"] == "critical" else "🟡"
        display_alerts.append({
            "icon": icon,
            "student_id": row["student_id"],
            "message": row["alert_message"],
            "severity": row["severity"],
            "score": row["engagement_score"],
            "risk": row["risk_level"],
        })
    return display_alerts


# ─── Standalone execution ────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add backend dir to sys.path for direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from dataset_generator import generate_dataset  # type: ignore[import-not-found]
    from feature_engineering import engineer_features, get_student_summary  # type: ignore[import-not-found]
    from ml_models import run_ml_pipeline  # type: ignore[import-not-found]

    df = generate_dataset()
    df = engineer_features(df)
    summary = get_student_summary(df)
    summary, _, _, _ = run_ml_pipeline(summary)

    alerts = generate_alerts(summary)
    print(f"\n📋 Sample Alerts:")
    for _, alert in alerts.head(5).iterrows():
        print(f"   {alert['alert_message']}")
