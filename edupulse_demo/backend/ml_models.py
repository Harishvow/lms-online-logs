"""
EduPulse – Machine Learning Models Module
K-Means Clustering and Random Forest Classification for
student engagement behavioral analysis and risk prediction.
"""

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.cluster import KMeans  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import classification_report, accuracy_score  # type: ignore
import warnings

warnings.filterwarnings("ignore")

# ─── Feature columns used for ML ─────────────────────────────────────────────
ML_FEATURES = [
    "avg_engagement_score",
    "total_video_time",
    "total_quiz_attempts",
    "avg_quiz_score",
    "total_assignments",
    "total_coding_subs",
    "avg_time_spent",
    "engagement_trend",
    "weekly_consistency",
]


def _prepare_features(summary_df: pd.DataFrame) -> tuple:
    """Prepare and scale feature matrix for ML models."""
    features = summary_df[ML_FEATURES].copy()
    features = features.fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    return X_scaled, scaler, features


# ═══════════════════════════════════════════════════════════════════════════════
# K-MEANS CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════

CLUSTER_LABELS = {
    0: "Highly Engaged",
    1: "Irregular",
    2: "At Risk / Drop-off",
}


def run_kmeans_clustering(summary_df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """
    Cluster students into engagement groups using K-Means.
    
    Clusters:
        Cluster 0 – Highly Engaged Students
        Cluster 1 – Irregular Students
        Cluster 2 – At Risk / Drop-off Students
    
    Returns:
        DataFrame with added 'cluster' and 'cluster_label' columns.
    """
    X_scaled, scaler, features = _prepare_features(summary_df)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    raw_labels = kmeans.fit_predict(X_scaled)

    # Reorder clusters so that the cluster with the highest mean engagement
    # score is labeled 0 (Highly Engaged), and lowest is labeled 2 (At Risk)
    cluster_means = {}
    for c in range(n_clusters):
        mask = raw_labels == c
        cluster_means[c] = features.loc[mask, "avg_engagement_score"].mean() if mask.any() else 0  # type: ignore

    sorted_clusters = sorted(cluster_means.keys(), key=lambda x: cluster_means[x], reverse=True)
    label_mapping = {old: new for new, old in enumerate(sorted_clusters)}

    summary_df = summary_df.copy()
    summary_df["cluster"] = [label_mapping[l] for l in raw_labels]
    summary_df["cluster_label"] = summary_df["cluster"].map(CLUSTER_LABELS)

    # Print summary
    print(f"\n🔬 K-Means Clustering Results ({n_clusters} clusters):")
    for c in range(n_clusters):
        count = (summary_df["cluster"] == c).sum()  # type: ignore
        avg = summary_df[summary_df["cluster"] == c]["avg_engagement_score"].mean()
        print(f"   {CLUSTER_LABELS[c]}: {count} students (avg score: {avg:.1f})")

    return summary_df


# ═══════════════════════════════════════════════════════════════════════════════
# RANDOM FOREST RISK CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

RISK_LABELS = {0: "Safe", 1: "Moderate Risk", 2: "High Risk"}


def _create_risk_labels(summary_df: pd.DataFrame) -> pd.Series:
    """
    Create training labels for risk classification based on engagement metrics.
    
    Logic:
        - High Risk: avg_engagement_score < 30 OR engagement_trend < -2.0
        - Moderate Risk: avg_engagement_score < 50 OR engagement_trend < -0.5
        - Safe: all others
    """
    conditions = [
        (summary_df["avg_engagement_score"] < 30) | (summary_df["engagement_trend"] < -2.0),
        (summary_df["avg_engagement_score"] < 50) | (summary_df["engagement_trend"] < -0.5),
    ]
    choices = [2, 1]  # High Risk, Moderate Risk
    return pd.Series(np.select(conditions, choices, default=0), name="risk_label")


def run_random_forest(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Train a Random Forest classifier to predict student risk levels.
    
    Risk Levels:
        0 – Safe
        1 – Moderate Risk
        2 – High Risk
    
    Returns:
        DataFrame with added 'risk_level' and 'risk_label' columns.
    """
    X_scaled, scaler, features = _prepare_features(summary_df)
    y = _create_risk_labels(summary_df)

    # Train-test split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)

    # Evaluate
    y_pred_test = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)

    print(f"\n🌲 Random Forest Classifier Results:")
    print(f"   Test Accuracy: {accuracy:.2%}")
    print(f"\n   Classification Report:")
    report = classification_report(y_test, y_pred_test, target_names=list(RISK_LABELS.values()))
    print(report)

    # Feature importance
    importance = pd.DataFrame({
        "feature": ML_FEATURES,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("   Top Feature Importances:")
    for _, row in importance.head(5).iterrows():
        print(f"     • {row['feature']}: {row['importance']:.3f}")

    # Predict on full dataset
    full_predictions = rf.predict(X_scaled)
    summary_df = summary_df.copy()
    summary_df["risk_level"] = full_predictions
    summary_df["risk_label"] = summary_df["risk_level"].map(RISK_LABELS)

    # Risk distribution
    print(f"\n   Risk Distribution:")
    for level, label in RISK_LABELS.items():
        count = (summary_df["risk_level"] == level).sum()  # type: ignore
        print(f"     {label}: {count} students")

    return summary_df, rf, importance


def run_ml_pipeline(summary_df: pd.DataFrame) -> tuple:
    """
    Run the complete ML analysis pipeline.
    
    Returns:
        (enhanced_df, model, feature_importance)
    """
    print("=" * 60)
    print("🧠 Running ML Analysis Pipeline")
    print("=" * 60)

    # Step 1: K-Means Clustering
    summary_df = run_kmeans_clustering(summary_df)

    # Step 2: Random Forest Risk Prediction
    summary_df, model, importance = run_random_forest(summary_df)

    # Generate ML-based report
    ml_report = _generate_ml_report(summary_df, importance)

    return summary_df, model, importance, ml_report


def _generate_ml_report(summary_df: pd.DataFrame, importance: pd.DataFrame) -> str:
    """Generate a human-readable ML analysis report."""
    total = len(summary_df)
    high_risk = summary_df[summary_df["risk_label"] == "High Risk"]
    moderate = summary_df[summary_df["risk_label"] == "Moderate Risk"]
    safe = summary_df[summary_df["risk_label"] == "Safe"]

    report = f"""## 🧠 Machine Learning Analysis Report

### Clustering Analysis (K-Means, k=3)

Students were grouped into 3 behavioral clusters:

| Cluster | Students | Avg Score | Description |
|---------|----------|-----------|-------------|
| Highly Engaged | {(summary_df['cluster']==0).sum()} | {summary_df[summary_df['cluster']==0]['avg_engagement_score'].mean():.1f} | Consistent high activity |  # type: ignore
| Irregular | {(summary_df['cluster']==1).sum()} | {summary_df[summary_df['cluster']==1]['avg_engagement_score'].mean():.1f} | Inconsistent participation |  # type: ignore
| At Risk | {(summary_df['cluster']==2).sum()} | {summary_df[summary_df['cluster']==2]['avg_engagement_score'].mean():.1f} | Low or declining engagement |  # type: ignore

### Risk Prediction (Random Forest)

| Risk Level | Count | Percentage |
|------------|-------|------------|
| ✅ Safe | {len(safe)} | {len(safe)/total*100:.0f}% |
| ⚡ Moderate Risk | {len(moderate)} | {len(moderate)/total*100:.0f}% |
| 🚨 High Risk | {len(high_risk)} | {len(high_risk)/total*100:.0f}% |

### Top Predictive Features

{chr(10).join(f"- **{row['feature']}**: {row['importance']:.3f}" for _, row in importance.head(5).iterrows())}

### High Risk Students Requiring Immediate Attention

{chr(10).join(f"- **Student {row['student_id']}** — Score: {row['avg_engagement_score']:.1f}, Trend: {row['engagement_trend']:.2f}" for _, row in high_risk.head(10).iterrows())}

---
*Analysis powered by scikit-learn ML Pipeline*
"""
    return report


# ─── Standalone execution ────────────────────────────────────────────────────
if __name__ == "__main__":
    from .dataset_generator import generate_dataset  # type: ignore
    from .feature_engineering import engineer_features, get_student_summary  # type: ignore

    df = generate_dataset()
    df = engineer_features(df)
    summary = get_student_summary(df)

    summary, model, importance, report = run_ml_pipeline(summary)
    print("\n" + report)
