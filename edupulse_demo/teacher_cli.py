#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║          EduPulse – AI-Powered Teacher Insights CLI                         ║
║          ─────────────────────────────────────────                          ║
║  Enter your Teacher ID to get LLM-analyzed insights on your students'      ║
║  online activity logs, engagement patterns, and early warning alerts.      ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Usage:
    python teacher_cli.py              # Interactive mode
    python teacher_cli.py TCH001       # Direct teacher ID
"""

import os
import sys
import json
import time
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv  # type: ignore

# ─── Load .env file ──────────────────────────────────────────────────────────
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_env_path)

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

from dataset_generator import generate_dataset  # type: ignore
from feature_engineering import engineer_features, get_student_summary  # type: ignore
from ml_models import run_ml_pipeline  # type: ignore
from alert_system import generate_alerts, get_alert_summary  # type: ignore
from teacher_student_mapping import (  # type: ignore
    load_mapping,
    get_teacher_info,
    list_all_teachers,
    generate_mapping,
    save_mapping,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

GEMINI_MODEL = "gemini-2.0-flash"

# The FIXED prompt template sent to the LLM for analyzing teacher-specific data
FIXED_LLM_PROMPT = """You are an expert educational data analyst specializing in Learning Management System (LMS) analytics.
A teacher has requested insights about their students' online activity logs.

═══════════════════════════════════════
TEACHER INFORMATION
═══════════════════════════════════════
Teacher: {teacher_name}
Teacher ID: {teacher_id}
Department: {department}
Subject: {subject}
Total Students Under This Teacher: {student_count}

═══════════════════════════════════════
STUDENT ONLINE ACTIVITY LOG DATA
═══════════════════════════════════════
{student_data_summary}

═══════════════════════════════════════
DETAILED PER-STUDENT METRICS
═══════════════════════════════════════
{per_student_details}

═══════════════════════════════════════
INSTRUCTIONS
═══════════════════════════════════════
Analyze the above student online activity log data and provide a STRUCTURED report with the following sections:

1. **📊 OVERALL CLASS HEALTH SUMMARY**
   - Overall engagement score (out of 100)
   - Class health rating (Excellent / Good / Needs Attention / Critical)
   - Key statistics in a concise format

2. **🏆 TOP PERFORMING STUDENTS** (Top 5)
   - List with student ID, engagement score, and what makes them stand out
   - Highlight their positive behaviors

3. **🚨 AT-RISK STUDENTS – IMMEDIATE ALERTS**
   - List students with critically low engagement (score < 30)
   - For each: student ID, engagement score, trend direction, specific concerns
   - Use ⚠️ WARNING and 🔴 CRITICAL markers

4. **📉 DECLINING ENGAGEMENT ALERTS**
   - Students showing week-over-week decline in engagement
   - Identify drop-off patterns
   - Include trend slopes and consistency metrics

5. **🔔 TEACHER ACTION ITEMS** (Prioritized)
   - Immediate actions (within 24 hours)
   - Short-term actions (this week)
   - Long-term strategies (this month)
   - Be specific: name student IDs and recommended interventions

6. **📈 ENGAGEMENT PATTERN ANALYSIS**
   - Cluster distribution of students
   - Common behavioral patterns observed
   - Video watch time vs quiz performance correlation insights

7. **💡 PERSONALIZED RECOMMENDATIONS**
   - Specific teaching strategy adjustments
   - Content delivery improvements
   - Student-specific intervention plans

Format the output with clear headers, bullet points, and use emojis for visual clarity.
Be data-driven: cite specific numbers, scores, and trends from the data.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# TERMINAL UI HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def print_banner():
    """Print the application banner."""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
 ╔═══════════════════════════════════════════════════════════════════════╗
 ║                                                                     ║
 ║     ███████╗██████╗ ██╗   ██╗██████╗ ██╗   ██╗██╗     ███████╗     ║
 ║     ██╔════╝██╔══██╗██║   ██║██╔══██╗██║   ██║██║     ██╔════╝     ║
 ║     █████╗  ██║  ██║██║   ██║██████╔╝██║   ██║██║     ███████╗     ║
 ║     ██╔══╝  ██║  ██║██║   ██║██╔═══╝ ██║   ██║██║     ╚════██║     ║
 ║     ███████╗██████╔╝╚██████╔╝██║     ╚██████╔╝███████╗███████║     ║
 ║     ╚══════╝╚═════╝  ╚═════╝ ╚═╝      ╚═════╝ ╚══════╝╚══════╝     ║
 ║                                                                     ║
 ║        🤖 AI-Powered Student Engagement Insights Engine 🤖          ║
 ║                                                                     ║
 ╚═══════════════════════════════════════════════════════════════════════╝
{Colors.END}
{Colors.DIM}  Online Activity Logs → LLM Analysis → Structured Insights & Alerts{Colors.END}
{Colors.DIM}  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}
"""
    print(banner)


def print_section(title: str, color: str = Colors.CYAN):
    """Print a section divider."""
    width = 70
    print(f"\n{color}{Colors.BOLD}{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}{Colors.END}\n")


def print_loading(message: str):
    """Print a loading animation."""
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    for i in range(15):
        sys.stdout.write(f"\r  {Colors.YELLOW}{frames[i % len(frames)]} {message}...{Colors.END}")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write(f"\r  {Colors.GREEN}✅ {message} — Done!{Colors.END}\n")


def print_alert_box(message: str, severity: str = "info"):
    """Print a formatted alert box."""
    if severity == "critical":
        color = Colors.RED
        icon = "🔴"
    elif severity == "warning":
        color = Colors.YELLOW
        icon = "🟡"
    else:
        color = Colors.CYAN
        icon = "🔵"
    
    print(f"  {color}┌{'─' * 66}┐{Colors.END}")
    # Wrap long messages
    words = message.split()
    line = f"  {icon} "
    lines = []
    for word in words:
        if len(line) + len(word) + 1 > 64:
            lines.append(line)
            line = f"     {word}"
        else:
            line += f" {word}"
    lines.append(line)
    for l in lines:
        print(f"  {color}│ {l:<64} │{Colors.END}")
    print(f"  {color}└{'─' * 66}┘{Colors.END}")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_process_data() -> tuple:
    """Load dataset, engineer features, run ML pipeline."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "data", "student_activity_logs.csv")
    
    # Load or generate dataset
    if os.path.exists(csv_path):
        print_loading("Loading student activity logs from CSV")
        df = pd.read_csv(csv_path)
    else:
        print_loading("Generating synthetic student activity dataset")
        df = generate_dataset(csv_path)
    
    # Feature engineering
    print_loading("Running feature engineering pipeline")
    df = engineer_features(df)
    
    # Student summary
    print_loading("Computing student summaries")
    summary = get_student_summary(df)
    
    # ML pipeline (K-Means + Random Forest)
    print_loading("Running ML analysis (K-Means + Random Forest)")
    summary, model, importance, ml_report = run_ml_pipeline(summary)
    
    # Generate alerts
    print_loading("Generating early warning alerts")
    alerts = generate_alerts(summary)
    
    return df, summary, alerts, ml_report


def get_teacher_students_data(
    teacher_info: dict,
    raw_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
) -> dict:
    """
    Extract and prepare data specific to a teacher's students.
    
    Returns a dict with:
     - teacher_info
     - student_summary_text (formatted for LLM)
     - per_student_details (formatted for LLM)
     - alerts for these students
     - raw data stats
    """
    student_ids = teacher_info["student_ids"]
    
    # Filter data for this teacher's students
    teacher_raw = raw_df[raw_df["student_id"].isin(student_ids)]
    teacher_summary = summary_df[summary_df["student_id"].isin(student_ids)]
    teacher_alerts = alerts_df[alerts_df["student_id"].isin(student_ids)] if not alerts_df.empty else pd.DataFrame()
    
    # Build summary statistics text
    avg_engagement = teacher_summary["avg_engagement_score"].mean()
    at_risk = teacher_summary[teacher_summary["avg_engagement_score"] < 30]
    high_perf = teacher_summary[teacher_summary["avg_engagement_score"] > 60]
    declining = teacher_summary[teacher_summary["engagement_trend"] < -1.0]
    
    summary_text = f"""
AGGREGATE STATISTICS:
  • Total Students: {len(teacher_summary)}
  • Average Engagement Score: {avg_engagement:.1f} / 100
  • High Performing Students (score > 60): {len(high_perf)}
  • At-Risk Students (score < 30): {len(at_risk)}
  • Students with Declining Trend: {len(declining)}
  • Average Video Watch Time: {teacher_summary['total_video_time'].mean():.0f} min (total across {12} weeks)
  • Average Quiz Score: {teacher_summary['avg_quiz_score'].mean():.1f}
  • Average Forum Posts: {teacher_summary['total_forum_posts'].mean():.1f} (total)
  • Average Time Spent: {teacher_summary['avg_time_spent'].mean():.1f} min/week

ENGAGEMENT DISTRIBUTION:
  • Score > 60 (High):     {len(teacher_summary[teacher_summary['avg_engagement_score'] > 60])} students
  • Score 30-60 (Medium):  {len(teacher_summary[(teacher_summary['avg_engagement_score'] >= 30) & (teacher_summary['avg_engagement_score'] <= 60)])} students
  • Score < 30 (Low):      {len(teacher_summary[teacher_summary['avg_engagement_score'] < 30])} students

RISK DISTRIBUTION:
  • Safe:          {len(teacher_summary[teacher_summary['risk_label'] == 'Safe'])} students
  • Moderate Risk: {len(teacher_summary[teacher_summary['risk_label'] == 'Moderate Risk'])} students
  • High Risk:     {len(teacher_summary[teacher_summary['risk_label'] == 'High Risk'])} students

CLUSTER DISTRIBUTION:
  • Highly Engaged:   {len(teacher_summary[teacher_summary['cluster_label'] == 'Highly Engaged'])} students
  • Irregular:        {len(teacher_summary[teacher_summary['cluster_label'] == 'Irregular'])} students
  • At Risk/Drop-off: {len(teacher_summary[teacher_summary['cluster_label'] == 'At Risk / Drop-off'])} students
"""
    
    # Build per-student details
    cols_to_show = [
        "student_id", "avg_engagement_score", "engagement_trend",
        "weekly_consistency", "avg_quiz_score", "total_video_time",
        "total_assignments", "total_late", "total_forum_posts",
        "avg_time_spent", "risk_label", "cluster_label",
    ]
    available_cols = [c for c in cols_to_show if c in teacher_summary.columns]
    per_student_text = teacher_summary[available_cols].to_string(index=False)
    
    return {
        "teacher_info": teacher_info,
        "summary_text": summary_text,
        "per_student_text": per_student_text,
        "teacher_summary_df": teacher_summary,
        "teacher_alerts_df": teacher_alerts,
        "teacher_raw_df": teacher_raw,
        "stats": {
            "total_students": len(teacher_summary),
            "avg_engagement": avg_engagement,
            "at_risk_count": len(at_risk),
            "high_perf_count": len(high_perf),
            "declining_count": len(declining),
            "critical_alerts": len(teacher_alerts[teacher_alerts["severity"] == "critical"]) if not teacher_alerts.empty else 0,
            "warning_alerts": len(teacher_alerts[teacher_alerts["severity"] == "warning"]) if not teacher_alerts.empty else 0,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LLM ANALYSIS (GEMINI)
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_with_gemini(teacher_data: dict) -> Optional[str]:
    """
    Send student activity data to Gemini LLM with the fixed prompt template.
    Returns the structured analysis text, or None if unavailable.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    
    # Validate API key is present and not a placeholder
    if not api_key or api_key in ("paste_your_api_key_here", "your-key-here", "YOUR_API_KEY"):
        print(f"\n  {Colors.YELLOW}⚠️  No valid GEMINI_API_KEY found.{Colors.END}")
        print(f"  {Colors.DIM}  ┌────────────────────────────────────────────────────────┐{Colors.END}")
        print(f"  {Colors.DIM}  │  To enable Gemini AI analysis:                        │{Colors.END}")
        print(f"  {Colors.DIM}  │  1. Get your free API key from:                       │{Colors.END}")
        print(f"  {Colors.CYAN}  │     https://aistudio.google.com/apikey                │{Colors.END}")
        print(f"  {Colors.DIM}  │  2. Open .env file in the project root                │{Colors.END}")
        print(f"  {Colors.DIM}  │  3. Replace 'paste_your_api_key_here' with your key   │{Colors.END}")
        print(f"  {Colors.DIM}  │  OR set it in terminal:                               │{Colors.END}")
        print(f"  {Colors.DIM}  │     export GEMINI_API_KEY='your-key-here'             │{Colors.END}")
        print(f"  {Colors.DIM}  └────────────────────────────────────────────────────────┘{Colors.END}")
        print(f"  {Colors.DIM}  Falling back to local analysis engine...{Colors.END}\n")
        return None
    
    print(f"  {Colors.GREEN}🔑 Gemini API key detected!{Colors.END}")
    
    try:
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=(
                "You are an expert educational data analyst. "
                "Provide structured, actionable insights about student engagement "
                "from LMS online activity logs. Use clear formatting with headers, "
                "bullet points, and emojis for readability."
            ),
        )
        
        info = teacher_data["teacher_info"]
        prompt = FIXED_LLM_PROMPT.format(
            teacher_name=info["name"],
            teacher_id=info["teacher_id"],
            department=info["department"],
            subject=info["subject"],
            student_count=info["student_count"],
            student_data_summary=teacher_data["summary_text"],
            per_student_details=teacher_data["per_student_text"],
        )
        
        print_loading("Sending data to Gemini AI for analysis")
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=4000,
                temperature=0.4,
            ),
        )
        
        if response.text and len(response.text) > 100:
            return response.text
        else:
            print(f"  {Colors.YELLOW}⚠️  LLM returned insufficient response.{Colors.END}")
            return None
            
    except ImportError:
        print(f"  {Colors.YELLOW}⚠️  google-generativeai package not installed.{Colors.END}")
        print(f"  {Colors.DIM}  Install with: pip install google-generativeai{Colors.END}")
        return None
    except Exception as e:
        print(f"  {Colors.RED}❌ LLM Error: {e}{Colors.END}")
        return None


def generate_local_analysis(teacher_data: dict) -> str:
    """
    Generate a structured analysis locally when LLM is unavailable.
    Uses the actual data to produce meaningful, data-driven insights.
    """
    info = teacher_data["teacher_info"]
    stats = teacher_data["stats"]
    t_summary = teacher_data["teacher_summary_df"]
    t_alerts = teacher_data["teacher_alerts_df"]
    
    # Determine class health
    avg = stats["avg_engagement"]
    if avg > 55:
        health = "Good ✅"
        health_desc = "Most students are actively engaged."
    elif avg > 40:
        health = "Needs Attention ⚡"
        health_desc = "Some students require intervention."
    elif avg > 25:
        health = "Concerning ⚠️"
        health_desc = "Significant portion of students are disengaged."
    else:
        health = "Critical 🚨"
        health_desc = "Majority of students are at risk."
    
    # Top performers
    top5 = t_summary.nlargest(5, "avg_engagement_score")
    top_lines = []
    for _, s in top5.iterrows():
        top_lines.append(
            f"   🏅 Student {int(s['student_id']):>3d}  │  Score: {s['avg_engagement_score']:>5.1f}  │  "
            f"Trend: {s['engagement_trend']:+.2f}  │  Quiz Avg: {s['avg_quiz_score']:.1f}"
        )
    
    # At-risk students
    at_risk = t_summary[t_summary["avg_engagement_score"] < 30].sort_values("avg_engagement_score")
    risk_lines = []
    for _, s in at_risk.iterrows():
        severity = "🔴 CRITICAL" if s["avg_engagement_score"] < 20 else "⚠️  WARNING"
        risk_lines.append(
            f"   {severity}  Student {int(s['student_id']):>3d}  │  Score: {s['avg_engagement_score']:>5.1f}  │  "
            f"Trend: {s['engagement_trend']:+.2f}  │  Risk: {s.get('risk_label', 'Unknown')}"
        )
    
    # Declining students
    declining = t_summary[t_summary["engagement_trend"] < -1.0].sort_values("engagement_trend")
    decline_lines = []
    for _, s in declining.iterrows():
        decline_lines.append(
            f"   📉 Student {int(s['student_id']):>3d}  │  Score: {s['avg_engagement_score']:>5.1f}  │  "
            f"Trend: {s['engagement_trend']:+.2f} /week  │  Consistency: {s['weekly_consistency']:.2f}"
        )
    
    # Alert details
    alert_lines = []
    if not t_alerts.empty:
        for _, a in t_alerts.head(10).iterrows():
            icon = "🔴" if a["severity"] == "critical" else "🟡"
            alert_lines.append(
                f"   {icon} Student {int(a['student_id']):>3d}  │  Score: {a['engagement_score']:>5.1f}  │  "
                f"{a['alert_reasons']}"
            )
    
    # Build the report
    report = f"""
{Colors.CYAN}{Colors.BOLD}
┌───────────────────────────────────────────────────────────────────────┐
│                 📊 OVERALL CLASS HEALTH SUMMARY                      │
└───────────────────────────────────────────────────────────────────────┘{Colors.END}

  Teacher:              {Colors.BOLD}{info['name']}{Colors.END} ({info['teacher_id']})
  Department:           {info['department']}
  Subject:              {info['subject']}
  Total Students:       {stats['total_students']}
  
  {Colors.BOLD}Class Health Rating:  {health}{Colors.END}
  {health_desc}
  
  ┌─────────────────────────────────────────────────┐
  │  Avg Engagement Score:  {Colors.BOLD}{avg:>6.1f}{Colors.END} / 100            │
  │  High Performers:       {Colors.GREEN}{stats['high_perf_count']:>6d}{Colors.END} students          │
  │  At-Risk Students:      {Colors.RED}{stats['at_risk_count']:>6d}{Colors.END} students          │
  │  Declining Trend:       {Colors.YELLOW}{stats['declining_count']:>6d}{Colors.END} students          │
  │  Critical Alerts:       {Colors.RED}{stats['critical_alerts']:>6d}{Colors.END}                   │
  │  Warning Alerts:        {Colors.YELLOW}{stats['warning_alerts']:>6d}{Colors.END}                   │
  └─────────────────────────────────────────────────┘

{Colors.GREEN}{Colors.BOLD}
┌───────────────────────────────────────────────────────────────────────┐
│                 🏆 TOP PERFORMING STUDENTS                           │
└───────────────────────────────────────────────────────────────────────┘{Colors.END}
{chr(10).join(top_lines) if top_lines else "   No high-performing students found."}

{Colors.RED}{Colors.BOLD}
┌───────────────────────────────────────────────────────────────────────┐
│                 🚨 AT-RISK STUDENTS – IMMEDIATE ALERTS               │
└───────────────────────────────────────────────────────────────────────┘{Colors.END}
{chr(10).join(risk_lines) if risk_lines else f"   {Colors.GREEN}✅ No students currently at critical risk level.{Colors.END}"}

{Colors.YELLOW}{Colors.BOLD}
┌───────────────────────────────────────────────────────────────────────┐
│                 📉 DECLINING ENGAGEMENT ALERTS                       │
└───────────────────────────────────────────────────────────────────────┘{Colors.END}
{chr(10).join(decline_lines) if decline_lines else f"   {Colors.GREEN}✅ No students showing significant engagement decline.{Colors.END}"}

{Colors.BLUE}{Colors.BOLD}
┌───────────────────────────────────────────────────────────────────────┐
│                 🔔 ACTIVE ALERTS FOR YOUR STUDENTS                   │
└───────────────────────────────────────────────────────────────────────┘{Colors.END}
{chr(10).join(alert_lines) if alert_lines else f"   {Colors.GREEN}✅ No active alerts at this time.{Colors.END}"}

{Colors.CYAN}{Colors.BOLD}
┌───────────────────────────────────────────────────────────────────────┐
│                 🔔 TEACHER ACTION ITEMS                              │
└───────────────────────────────────────────────────────────────────────┘{Colors.END}
"""
    
    # Generate action items based on data
    actions = []
    if stats["at_risk_count"] > 0:
        critical_ids = at_risk["student_id"].tolist()[:5]
        id_str = ", ".join([str(int(x)) for x in critical_ids])
        actions.append(f"  {Colors.RED}🔴 IMMEDIATE{Colors.END}: Reach out to students [{id_str}] within 24 hours")  # type: ignore
        actions.append(f"     — Schedule one-on-one meetings to understand their challenges")
    
    if stats["declining_count"] > 0:
        decline_ids = declining["student_id"].tolist()[:5]
        id_str = ", ".join([str(int(x)) for x in decline_ids])
        actions.append(f"  {Colors.YELLOW}🟡 THIS WEEK{Colors.END}: Monitor students [{id_str}] showing decline")  # type: ignore
        actions.append(f"     — Set up weekly check-in reminders for these students")
    
    moderate_risk = t_summary[t_summary.get("risk_label", pd.Series()) == "Moderate Risk"] if "risk_label" in t_summary.columns else pd.DataFrame()  # type: ignore
    if not moderate_risk.empty:
        actions.append(f"  {Colors.CYAN}🔵 THIS MONTH{Colors.END}: Implement peer mentoring for {len(moderate_risk)} moderate-risk students")  # type: ignore
        actions.append(f"     — Pair them with top performers for study groups")
    
    actions.append(f"  {Colors.GREEN}💡 ONGOING{Colors.END}: Review video content for engagement optimization")  # type: ignore
    actions.append(f"     — Consider breaking long videos into 10-15 min segments")
    actions.append(f"     — Add interactive quizzes within video content")
    
    report += "\n".join(actions)
    
    # Engagement pattern analysis
    report += f"""

{Colors.CYAN}{Colors.BOLD}
┌───────────────────────────────────────────────────────────────────────┐
│                 📈 ENGAGEMENT PATTERN ANALYSIS                       │
└───────────────────────────────────────────────────────────────────────┘{Colors.END}

  Student Cluster Distribution:
"""
    if "cluster_label" in t_summary.columns:
        for label in ["Highly Engaged", "Irregular", "At Risk / Drop-off"]:
            count = len(t_summary[t_summary["cluster_label"] == label])
            pct = count / len(t_summary) * 100
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            report += f"    {label:<20s} │ {bar} │ {count:>2d} ({pct:.0f}%)\n"
    
    # Key observations
    avg_video = t_summary["total_video_time"].mean()
    avg_quiz = t_summary["avg_quiz_score"].mean()
    avg_forum = t_summary["total_forum_posts"].mean()
    
    report += f"""
  Key Observations:
    • Average total video watch time: {avg_video:.0f} minutes
    • Average quiz score: {avg_quiz:.1f}%
    • Average forum participation: {avg_forum:.1f} posts
    • Students with high video time tend to score {'better' if avg_video > 500 else 'average'} on quizzes
    • Forum participation is {'strong' if avg_forum > 20 else 'low'} — consider engagement incentives
"""

    report += f"""
{Colors.DIM}───────────────────────────────────────────────────────────────────────
  Analysis Source: EduPulse Local Analysis Engine (ML + Rule-Based)
  Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  💡 Tip: Set GEMINI_API_KEY for enhanced AI-powered insights
───────────────────────────────────────────────────────────────────────{Colors.END}
"""
    
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CLI FLOW
# ═══════════════════════════════════════════════════════════════════════════════

def display_teacher_insights(teacher_id: str, raw_df, summary_df, alerts_df):
    """Main function to display full insights for a teacher."""
    
    # Load teacher mapping
    mapping = load_mapping()
    teacher_info = get_teacher_info(teacher_id, mapping)
    
    if not teacher_info:
        print(f"\n  {Colors.RED}❌ Teacher ID '{teacher_id}' not found!{Colors.END}")
        print(f"  {Colors.DIM}  Available IDs: {', '.join(mapping.keys())}{Colors.END}")
        return False
    
    print_section(f"🎓 Teacher: {teacher_info['name']} ({teacher_id.upper()})")
    
    # Get teacher-specific data
    print_loading("Extracting teacher-specific student data")
    teacher_data = get_teacher_students_data(teacher_info, raw_df, summary_df, alerts_df)
    
    # Print quick stats
    stats = teacher_data["stats"]
    print(f"\n  {Colors.BOLD}Quick Overview:{Colors.END}")
    print(f"  ├── Students: {stats['total_students']}")
    print(f"  ├── Avg Engagement: {stats['avg_engagement']:.1f}")
    print(f"  ├── At Risk: {Colors.RED}{stats['at_risk_count']}{Colors.END}")
    print(f"  ├── Declining: {Colors.YELLOW}{stats['declining_count']}{Colors.END}")
    print(f"  └── Alerts: {Colors.RED}{stats['critical_alerts']} critical{Colors.END}, {Colors.YELLOW}{stats['warning_alerts']} warning{Colors.END}")
    
    # Try LLM analysis first, fallback to local
    print_section("🤖 AI Analysis Engine")
    
    llm_result = analyze_with_gemini(teacher_data)
    
    if llm_result:
        print(f"\n  {Colors.GREEN}✅ Gemini AI Analysis Complete{Colors.END}")
        print(f"  {Colors.DIM}Source: Google Gemini ({GEMINI_MODEL}){Colors.END}\n")
        print(llm_result)
    else:
        print(f"  {Colors.YELLOW}⚡ Using Local Analysis Engine (ML + Rules){Colors.END}\n")
        local_report = generate_local_analysis(teacher_data)
        print(local_report)
    
    return True


def interactive_mode():
    """Run in interactive terminal mode."""
    print_banner()
    
    # Load and process data
    print_section("📦 DATA LOADING & PROCESSING")
    raw_df, summary_df, alerts_df, ml_report = load_and_process_data()
    
    # Ensure teacher mapping exists
    mapping = load_mapping()
    
    print(f"\n  {Colors.GREEN}✅ System ready! {len(summary_df)} students loaded across {len(mapping)} teachers.{Colors.END}")
    
    # Show available teachers
    list_all_teachers(mapping)
    
    while True:
        print(f"\n{Colors.BOLD}{'─' * 70}{Colors.END}")
        teacher_id = input(f"\n  {Colors.CYAN}📝 Enter Teacher ID (e.g., TCH001) or 'quit' to exit: {Colors.END}").strip()
        
        if teacher_id.lower() in ("quit", "exit", "q"):
            print(f"\n  {Colors.GREEN}👋 Thank you for using EduPulse! Goodbye.{Colors.END}\n")
            break
        
        if teacher_id.lower() == "list":
            list_all_teachers(mapping)
            continue
        
        if teacher_id.lower() == "help":
            print(f"""
  {Colors.BOLD}Available Commands:{Colors.END}
    TCH001-TCH010  — Enter a teacher ID to view insights
    list           — Show all registered teachers
    help           — Show this help message
    quit           — Exit the application
""")
            continue
        
        display_teacher_insights(teacher_id, raw_df, summary_df, alerts_df)


def main():
    """Entry point."""
    if len(sys.argv) > 1:
        # Direct mode: python teacher_cli.py TCH001
        teacher_id = sys.argv[1]
        print_banner()
        
        print_section("📦 DATA LOADING & PROCESSING")
        raw_df, summary_df, alerts_df, ml_report = load_and_process_data()
        
        display_teacher_insights(teacher_id, raw_df, summary_df, alerts_df)
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()
