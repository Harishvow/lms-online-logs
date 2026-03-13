"""
EduPulse – AI Powered Student Engagement Early Warning System
Interactive Streamlit Dashboard
"""

import sys
import os

# ─── Path Setup ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backend.dataset_generator import generate_dataset
from backend.feature_engineering import engineer_features, get_student_summary
from backend.llm_analysis import get_llm_or_fallback
from backend.ml_models import run_ml_pipeline
from backend.alert_system import generate_alerts, format_alerts_for_display

# ─── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduPulse – AI Student Engagement",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        color: white;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1.05rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.15);
    }
    .metric-value {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #888;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }

    /* Alert Cards */
    .alert-critical {
        background: linear-gradient(135deg, rgba(255,71,87,0.15), rgba(255,71,87,0.05));
        border-left: 4px solid #ff4757;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        font-size: 0.9rem;
    }
    .alert-warning {
        background: linear-gradient(135deg, rgba(255,165,0,0.15), rgba(255,165,0,0.05));
        border-left: 4px solid #ffa500;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        font-size: 0.9rem;
    }

    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
    }

    /* Analysis box */
    .analysis-box {
        background: linear-gradient(135deg, rgba(102,126,234,0.08), rgba(118,75,162,0.05));
        border: 1px solid rgba(102,126,234,0.2);
        border-radius: 12px;
        padding: 1.5rem;
        line-height: 1.7;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Table styling */
    .dataframe {
        font-size: 0.85rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading & Caching ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_process_data():
    """Generate dataset, engineer features, and run full analysis pipeline."""
    # Step 1: Generate dataset
    csv_path = os.path.join(BASE_DIR, "data", "student_activity_logs.csv")
    df = generate_dataset(csv_path)

    # Step 2: Feature engineering
    df = engineer_features(df)
    summary = get_student_summary(df)

    # Step 3: LLM Analysis (with fallback)
    llm_result = get_llm_or_fallback(summary)

    # Step 4: ML Pipeline
    summary, model, importance, ml_report = run_ml_pipeline(summary)

    # Step 5: Generate alerts
    alerts = generate_alerts(summary)

    return df, summary, llm_result, ml_report, importance, alerts


# ─── Load Data ───────────────────────────────────────────────────────────────
with st.spinner("🚀 Initializing EduPulse AI Engine..."):
    df, summary, llm_result, ml_report, importance, alerts = load_and_process_data()


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎓 EduPulse – AI Powered Student Engagement</h1>
    <p>Early Warning System • Real-time Analytics • Intelligent Insights</p>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Dashboard Controls")
    st.markdown("---")

    # View selector
    view_mode = st.selectbox(
        "📊 Dashboard View",
        ["Overview", "Student Explorer", "Risk Analysis", "AI Insights"],
        index=0,
    )

    st.markdown("---")

    # Filters
    st.markdown("### 🔍 Filters")

    risk_filter = st.multiselect(
        "Risk Level",
        options=["Safe", "Moderate Risk", "High Risk"],
        default=["Safe", "Moderate Risk", "High Risk"],
    )

    cluster_filter = st.multiselect(
        "Cluster",
        options=["Highly Engaged", "Irregular", "At Risk / Drop-off"],
        default=["Highly Engaged", "Irregular", "At Risk / Drop-off"],
    )

    score_range = st.slider(
        "Engagement Score Range",
        min_value=0.0,
        max_value=float(summary["avg_engagement_score"].max()) + 10,
        value=(0.0, float(summary["avg_engagement_score"].max()) + 10),
        step=5.0,
    )

    st.markdown("---")
    st.markdown("### 📈 Quick Stats")
    st.metric("Total Students", len(summary))
    st.metric("Avg Engagement", f"{summary['avg_engagement_score'].mean():.1f}")
    st.metric("Active Alerts", len(alerts))

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.8rem;'>"
        "Powered by EduPulse AI Engine<br>v1.0.0</div>",
        unsafe_allow_html=True,
    )

# Apply filters
filtered = summary[
    (summary["risk_label"].isin(risk_filter)) &
    (summary["cluster_label"].isin(cluster_filter)) &
    (summary["avg_engagement_score"] >= score_range[0]) &
    (summary["avg_engagement_score"] <= score_range[1])
]


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if view_mode == "Overview":

    # ─── KPI Metrics ─────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(summary)}</div>
            <div class="metric-label">Total Students</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        avg_score = summary["avg_engagement_score"].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_score:.1f}</div>
            <div class="metric-label">Avg Engagement Score</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        at_risk_count = (summary["risk_label"] == "High Risk").sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="background: linear-gradient(135deg, #ff4757, #ff6b81); -webkit-background-clip: text;">{at_risk_count}</div>
            <div class="metric-label">Students At Risk</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        alert_count = len(alerts[alerts["severity"] == "critical"]) if not alerts.empty else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="background: linear-gradient(135deg, #ffa502, #ff6348); -webkit-background-clip: text;">{alert_count}</div>
            <div class="metric-label">Critical Alerts</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Charts Row 1 ───────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header">📊 Engagement Score Distribution</div>', unsafe_allow_html=True)
        fig_hist = px.histogram(
            filtered,
            x="avg_engagement_score",
            nbins=25,
            color_discrete_sequence=["#667eea"],
            labels={"avg_engagement_score": "Engagement Score"},
        )
        fig_hist.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            height=350,
            margin=dict(t=20, b=40, l=40, r=20),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Count"),
        )
        fig_hist.update_traces(
            marker=dict(
                line=dict(width=1, color="rgba(255,255,255,0.2)"),
                opacity=0.85,
            )
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">🎯 Student Cluster Distribution</div>', unsafe_allow_html=True)
        cluster_counts = filtered["cluster_label"].value_counts().reset_index()
        cluster_counts.columns = ["Cluster", "Count"]
        fig_pie = px.pie(
            cluster_counts,
            values="Count",
            names="Cluster",
            color="Cluster",
            color_discrete_map={
                "Highly Engaged": "#2ed573",
                "Irregular": "#ffa502",
                "At Risk / Drop-off": "#ff4757",
            },
            hole=0.45,
        )
        fig_pie.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=350,
            margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(orientation="h", y=-0.1),
        )
        fig_pie.update_traces(
            textposition="inside",
            textinfo="percent+label",
            textfont_size=12,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ─── Weekly Engagement Trend ─────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Weekly Engagement Trend</div>', unsafe_allow_html=True)

    weekly_trend = df.groupby(["week_number", "engagement_category"]).agg(
        avg_score=("engagement_score", "mean"),
    ).reset_index()

    fig_trend = px.line(
        weekly_trend,
        x="week_number",
        y="avg_score",
        color="engagement_category",
        color_discrete_map={
            "high_engagement": "#2ed573",
            "moderate_engagement": "#ffa502",
            "low_engagement": "#ff4757",
            "dropoff_behavior": "#a55eea",
        },
        labels={"week_number": "Week", "avg_score": "Avg Engagement Score", "engagement_category": "Category"},
        markers=True,
    )
    fig_trend.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
        margin=dict(t=20, b=40, l=40, r=20),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", dtick=1),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        legend=dict(orientation="h", y=-0.15),
    )
    fig_trend.update_traces(line=dict(width=3))
    st.plotly_chart(fig_trend, use_container_width=True)

    # ─── Risk Alert Panel ────────────────────────────────────────────────
    st.markdown('<div class="section-header">🚨 Risk Alert Panel</div>', unsafe_allow_html=True)

    if not alerts.empty:
        display_alerts = format_alerts_for_display(alerts)
        for alert in display_alerts[:10]:
            css_class = "alert-critical" if alert["severity"] == "critical" else "alert-warning"
            st.markdown(
                f'<div class="{css_class}">'
                f'{alert["icon"]} <strong>Student {alert["student_id"]}</strong> – '
                f'{alert["risk"]} | Score: {alert["score"]:.1f}<br>'
                f'<span style="font-size: 0.8rem; opacity: 0.8;">{alert["message"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.success("✅ No critical alerts at this time.")


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW: STUDENT EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif view_mode == "Student Explorer":

    st.markdown('<div class="section-header">🔍 Student Explorer</div>', unsafe_allow_html=True)

    # Student selector
    selected_student = st.selectbox(
        "Select a Student",
        options=sorted(filtered["student_id"].unique()),
        format_func=lambda x: f"Student {x}",
    )

    if selected_student:
        student_data = summary[summary["student_id"] == selected_student].iloc[0]
        student_weekly = df[df["student_id"] == selected_student].sort_values("week_number")

        # Student profile cards
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Engagement Score", f"{student_data['avg_engagement_score']:.1f}")
        with c2:
            st.metric("Risk Level", student_data.get("risk_label", "N/A"))
        with c3:
            st.metric("Cluster", student_data.get("cluster_label", "N/A"))
        with c4:
            st.metric("Trend", f"{student_data['engagement_trend']:+.2f}")
        with c5:
            st.metric("Consistency", f"{student_data['weekly_consistency']:.2f}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Weekly engagement chart
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("##### 📈 Weekly Engagement Over Time")
            fig_student = go.Figure()
            fig_student.add_trace(go.Scatter(
                x=student_weekly["week_number"],
                y=student_weekly["engagement_score"],
                mode="lines+markers",
                line=dict(color="#667eea", width=3),
                marker=dict(size=8),
                name="Engagement Score",
                fill="tozeroy",
                fillcolor="rgba(102,126,234,0.1)",
            ))
            fig_student.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=300,
                margin=dict(t=10, b=30, l=40, r=20),
                xaxis=dict(title="Week", gridcolor="rgba(255,255,255,0.05)", dtick=1),
                yaxis=dict(title="Score", gridcolor="rgba(255,255,255,0.05)"),
            )
            st.plotly_chart(fig_student, use_container_width=True)

        with col_b:
            st.markdown("##### 📊 Activity Breakdown")
            activities = {
                "Video Time": student_data["total_video_time"],
                "Quiz Attempts": student_data["total_quiz_attempts"] * 10,
                "Assignments": student_data["total_assignments"] * 15,
                "Coding Subs": student_data["total_coding_subs"] * 15,
                "Forum Posts": student_data["total_forum_posts"] * 5,
            }
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=list(activities.values()),
                theta=list(activities.keys()),
                fill="toself",
                fillcolor="rgba(102,126,234,0.2)",
                line=dict(color="#667eea", width=2),
            ))
            fig_radar.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                height=300,
                margin=dict(t=30, b=30, l=60, r=60),
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                ),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # Weekly details table
        st.markdown("##### 📋 Weekly Activity Log")
        display_cols = [
            "week_number", "video_watch_time_minutes", "quiz_attempts",
            "quiz_score", "assignment_submission_count", "assignment_late_count",
            "coding_assignment_submissions", "coding_success_rate",
            "discussion_forum_posts", "total_time_spent_minutes", "engagement_score",
        ]
        st.dataframe(
            student_weekly[display_cols].reset_index(drop=True),
            use_container_width=True,
            height=350,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW: RISK ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif view_mode == "Risk Analysis":

    st.markdown('<div class="section-header">⚡ Risk Analysis Dashboard</div>', unsafe_allow_html=True)

    # Risk distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 🎯 Risk Level Distribution")
        risk_counts = filtered["risk_label"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        fig_risk = px.bar(
            risk_counts,
            x="Risk Level",
            y="Count",
            color="Risk Level",
            color_discrete_map={
                "Safe": "#2ed573",
                "Moderate Risk": "#ffa502",
                "High Risk": "#ff4757",
            },
        )
        fig_risk.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            height=350,
            margin=dict(t=20, b=40, l=40, r=20),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        fig_risk.update_traces(
            marker=dict(line=dict(width=0), opacity=0.9),
            texttemplate="%{y}",
            textposition="outside",
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    with col2:
        st.markdown("##### 📊 Engagement vs Risk Scatter")
        color_map = {"Safe": "#2ed573", "Moderate Risk": "#ffa502", "High Risk": "#ff4757"}
        fig_scatter = px.scatter(
            filtered,
            x="avg_engagement_score",
            y="engagement_trend",
            color="risk_label",
            size="avg_time_spent",
            color_discrete_map=color_map,
            labels={
                "avg_engagement_score": "Engagement Score",
                "engagement_trend": "Engagement Trend",
                "risk_label": "Risk Level",
            },
            hover_data=["student_id"],
        )
        fig_scatter.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=350,
            margin=dict(t=20, b=40, l=40, r=20),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Feature Importance
    st.markdown("##### 🌲 Random Forest Feature Importance")
    fig_imp = px.bar(
        importance.sort_values("importance", ascending=True),
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Viridis",
    )
    fig_imp.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=350,
        margin=dict(t=20, b=40, l=150, r=20),
        xaxis=dict(title="Importance", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title=""),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # Student Table
    st.markdown('<div class="section-header">📋 Complete Student Table</div>', unsafe_allow_html=True)

    display_df = filtered[[
        "student_id", "avg_engagement_score", "cluster_label",
        "risk_label", "avg_time_spent", "engagement_trend",
        "weekly_consistency",
    ]].copy()
    display_df.columns = [
        "Student ID", "Engagement Score", "Cluster",
        "Risk Level", "Avg Time (min)", "Trend", "Consistency",
    ]

    # Add alert status
    alerted_ids = set(alerts["student_id"].tolist()) if not alerts.empty else set()
    display_df["Alert Status"] = display_df["Student ID"].apply(
        lambda x: "🚨 Active" if x in alerted_ids else "✅ Clear"
    )

    st.dataframe(
        display_df.sort_values("Engagement Score"),
        use_container_width=True,
        height=500,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW: AI INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif view_mode == "AI Insights":

    st.markdown('<div class="section-header">🤖 AI-Powered Analysis</div>', unsafe_allow_html=True)

    # Source indicator
    if llm_result and llm_result.get("success"):
        source = llm_result.get("source", "Unknown")
        st.info(f"📡 Analysis Source: **{source}**")

        st.markdown(
            f'<div class="analysis-box">{llm_result["analysis"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("⚠️ LLM analysis unavailable. Displaying ML-based analysis.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ML Report (always shown)
    st.markdown('<div class="section-header">🧠 Machine Learning Analysis Report</div>', unsafe_allow_html=True)
    st.markdown(ml_report)

    # Heatmap: Student engagement over weeks
    st.markdown('<div class="section-header">🔥 Engagement Heatmap (Top 30 Students by Variance)</div>', unsafe_allow_html=True)

    # Select students with most variance in engagement
    student_variance = df.groupby("student_id")["engagement_score"].std().nlargest(30)
    heatmap_students = student_variance.index.tolist()
    heatmap_data = df[df["student_id"].isin(heatmap_students)].pivot_table(
        index="student_id",
        columns="week_number",
        values="engagement_score",
        aggfunc="mean",
    )

    fig_heat = px.imshow(
        heatmap_data,
        labels=dict(x="Week", y="Student ID", color="Score"),
        color_continuous_scale="RdYlGn",
        aspect="auto",
    )
    fig_heat.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        height=500,
        margin=dict(t=20, b=40, l=60, r=20),
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.8rem; padding: 1rem;'>"
    "🎓 EduPulse – AI Powered Student Engagement Early Warning System | "
    "Built with Streamlit, scikit-learn, and Plotly | "
    "© 2026 EduPulse"
    "</div>",
    unsafe_allow_html=True,
)
