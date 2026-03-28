"""
EduPulse – Feature Engineering Module
Computes derived engagement metrics from raw student activity logs.
"""

import pandas as pd  # type: ignore
import numpy as np  # type: ignore


def compute_engagement_score(df: pd.DataFrame) -> pd.Series:
    """
    Compute the composite engagement score per record.
    
    Formula:
        engagement_score = (video_watch_time_minutes × 0.3)
                         + (quiz_attempts × 2)
                         + (assignment_submission_count × 3)
                         + (coding_assignment_submissions × 3)
                         + (discussion_forum_posts × 1)
                         - (assignment_late_count × 2)
    """
    score = (
        df["video_watch_time_minutes"] * 0.3
        + df["quiz_attempts"] * 2
        + df["assignment_submission_count"] * 3
        + df["coding_assignment_submissions"] * 3
        + df["discussion_forum_posts"] * 1
        - df["assignment_late_count"] * 2
    )
    return score.round(2)


def compute_engagement_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-student engagement trend (slope of engagement over weeks).
    A negative slope indicates declining engagement.
    """
    trends = []
    for sid, group in df.groupby("student_id"):
        group = group.sort_values("week_number")
        if len(group) >= 2:
            x = group["week_number"].values
            y = group["engagement_score"].values
            # Linear regression slope
            slope = np.polyfit(x, y, 1)[0]
        else:
            slope = 0.0
        trends.append({"student_id": sid, "engagement_trend": round(float(slope), 3)})  # type: ignore
    return pd.DataFrame(trends)


def compute_weekly_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute weekly consistency as the coefficient of variation (CV)
    of engagement_score per student. Lower CV = more consistent.
    """
    consistency = (
        df.groupby("student_id")["engagement_score"]
        .agg(["mean", "std"])
        .reset_index()
    )
    consistency["weekly_consistency"] = np.where(
        consistency["mean"] > 0,
        1 - (consistency["std"] / consistency["mean"]).clip(0, 1),
        0
    )
    consistency["weekly_consistency"] = consistency["weekly_consistency"].round(3)
    return consistency[["student_id", "weekly_consistency"]]


def compute_average_time_spent(df: pd.DataFrame) -> pd.DataFrame:
    """Compute average time spent per student across all weeks."""
    avg = (
        df.groupby("student_id")["total_time_spent_minutes"]
        .mean()
        .reset_index()
        .rename(columns={"total_time_spent_minutes": "average_time_spent"})
    )
    avg["average_time_spent"] = avg["average_time_spent"].round(1)
    return avg


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    
    Adds:
        - engagement_score (per record)
        - engagement_trend (per student, merged)
        - weekly_consistency (per student, merged)
        - average_time_spent (per student, merged)
    
    Returns:
        Enhanced DataFrame with all derived features.
    """
    # 1. Engagement Score (per row)
    df = df.copy()
    df["engagement_score"] = compute_engagement_score(df)

    # 2. Per-student aggregated features
    trend_df = compute_engagement_trend(df)
    consistency_df = compute_weekly_consistency(df)
    avg_time_df = compute_average_time_spent(df)

    # 3. Merge per-student features back
    df = df.merge(trend_df, on="student_id", how="left")
    df = df.merge(consistency_df, on="student_id", how="left")
    df = df.merge(avg_time_df, on="student_id", how="left")

    print(f"✅ Feature engineering complete. Added columns: engagement_score, engagement_trend, weekly_consistency, average_time_spent")
    return df


def get_student_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary table with one row per student, aggregating
    their metrics across all weeks.
    """
    summary = df.groupby("student_id").agg(
        avg_engagement_score=("engagement_score", "mean"),
        total_video_time=("video_watch_time_minutes", "sum"),
        total_quiz_attempts=("quiz_attempts", "sum"),
        avg_quiz_score=("quiz_score", "mean"),
        total_assignments=("assignment_submission_count", "sum"),
        total_late=("assignment_late_count", "sum"),
        total_coding_subs=("coding_assignment_submissions", "sum"),
        avg_coding_rate=("coding_success_rate", "mean"),
        total_forum_posts=("discussion_forum_posts", "sum"),
        avg_time_spent=("total_time_spent_minutes", "mean"),
        engagement_trend=("engagement_trend", "first"),
        weekly_consistency=("weekly_consistency", "first"),
        engagement_category=("engagement_category", "first"),
    ).reset_index()

    # Round numeric columns
    numeric_cols = summary.select_dtypes(include=[np.number]).columns
    summary[numeric_cols] = summary[numeric_cols].round(2)

    return summary


# ─── Standalone execution ────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "data", "student_activity_logs.csv")

    if not os.path.exists(csv_path):
        print("❌ Dataset not found. Run dataset_generator.py first.")
    else:
        df = pd.read_csv(csv_path)
        df = engineer_features(df)
        summary = get_student_summary(df)
        print(f"\n📊 Student Summary ({len(summary)} students):")
        print(summary.head(10).to_string(index=False))
