# 🎓 EduPulse – AI Powered Student Engagement Early Warning System

<p align="center">
  <strong>An intelligent system that analyzes student online learning activity logs<br>
  and detects low engagement patterns using AI + Machine Learning.</strong>
</p>

---

## 🌟 Features

| Feature | Description |
|---------|-------------|
| **Synthetic Data Generation** | 200 students × 12 weeks of realistic LMS activity logs |
| **Feature Engineering** | Engagement score, trend analysis, weekly consistency metrics |
| **LLM Analysis** | AI-powered behavioral pattern detection with human-readable insights |
| **ML Pipeline** | K-Means clustering + Random Forest risk prediction |
| **IF-ELSE Fallback** | Automatic fallback from LLM to ML analysis |
| **Interactive Dashboard** | Streamlit dashboard with 4 views and rich visualizations |
| **Automated Alerts** | Selenium-powered teacher notification system |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    EduPulse AI Engine                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────────┐   │
│  │ Dataset   │──▶│  Feature     │──▶│  LLM Analysis    │   │
│  │ Generator │   │  Engineering │   │  (Gemini)        │   │
│  └──────────┘   └──────────────┘   └────────┬─────────┘   │
│                                              │              │
│                                    ┌─────────▼─────────┐   │
│                                    │  IF LLM succeeds  │   │
│                                    │  → Use LLM        │   │
│                                    │  ELSE              │   │
│                                    │  → Use ML Pipeline │   │
│                                    └─────────┬─────────┘   │
│                                              │              │
│                        ┌─────────────────────┼──────┐      │
│                        │                     │      │      │
│                  ┌─────▼─────┐  ┌────────────▼──┐  │      │
│                  │  K-Means  │  │ Random Forest │  │      │
│                  │ Clustering│  │  Classifier   │  │      │
│                  └─────┬─────┘  └────────┬──────┘  │      │
│                        └────────┬────────┘         │      │
│                                 │                  │      │
│                        ┌────────▼────────┐         │      │
│                        │  Alert System   │─────────┘      │
│                        └────────┬────────┘                │
│                                 │                          │
│              ┌──────────────────┼──────────────────┐      │
│              │                  │                  │      │
│        ┌─────▼──────┐  ┌───────▼───────┐  ┌──────▼───┐  │
│        │  Streamlit  │  │   Selenium    │  │  Alert   │  │
│        │  Dashboard  │  │   Alert Bot   │  │  Logs    │  │
│        └────────────┘  └───────────────┘  └──────────┘  │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
edupulse_demo/
│
├── data/
│   └── student_activity_logs.csv       # Generated dataset
│
├── backend/
│   ├── __init__.py
│   ├── dataset_generator.py            # Synthetic data generation
│   ├── feature_engineering.py          # Derived metrics computation
│   ├── llm_analysis.py                 # LLM integration + fallback
│   ├── ml_models.py                    # K-Means + Random Forest
│   └── alert_system.py                 # Alert generation & management
│
├── dashboard/
│   ├── __init__.py
│   └── streamlit_app.py               # Interactive Streamlit dashboard
│
├── automation/
│   ├── __init__.py
│   └── selenium_alert_bot.py          # Selenium teacher alert bot
│
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.9+
- pip

### 2. Install Dependencies

```bash
cd edupulse_demo
pip install -r requirements.txt
```

### 3. Generate Dataset

```bash
python backend/dataset_generator.py
```

This creates `data/student_activity_logs.csv` with 200 students × 12 weeks = 2,400 records.

### 4. Run the Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

The dashboard will automatically:
1. Generate the dataset (if not exists)
2. Run feature engineering
3. Perform LLM analysis (with fallback)
4. Execute ML pipeline (K-Means + Random Forest)
5. Generate alerts
6. Display interactive visualizations

### 5. Run Selenium Alert Bot (Optional)

```bash
python automation/selenium_alert_bot.py
```

## 📊 Dashboard Views

### 1. Overview
- KPI metric cards (Total Students, Avg Score, At-Risk, Critical Alerts)
- Engagement Score Distribution (histogram)
- Student Cluster Distribution (donut chart)
- Weekly Engagement Trend (multi-line chart)
- Risk Alert Panel

### 2. Student Explorer
- Individual student drill-down
- Weekly engagement timeline
- Activity radar chart
- Detailed weekly activity log

### 3. Risk Analysis
- Risk level distribution (bar chart)
- Engagement vs Risk scatter plot
- Random Forest feature importance
- Complete student table with alert status

### 4. AI Insights
- LLM-generated analysis report
- ML pipeline analysis report
- Engagement heatmap (top 30 most variable students)

## 🧠 Machine Learning Models

### K-Means Clustering (k=3)
| Cluster | Label | Description |
|---------|-------|-------------|
| 0 | Highly Engaged | Consistent high activity across all metrics |
| 1 | Irregular | Inconsistent participation patterns |
| 2 | At Risk / Drop-off | Low or declining engagement |

### Random Forest Classifier
| Risk Level | Criteria |
|------------|----------|
| Safe | Score ≥ 50 AND Trend ≥ -0.5 |
| Moderate Risk | Score < 50 OR Trend < -0.5 |
| High Risk | Score < 30 OR Trend < -2.0 |

## 🔔 Alert System

Alerts are triggered when:
- `engagement_score < 30` (low engagement threshold)
- `risk_label == "High Risk"` (ML prediction)
- `engagement_trend < -1.5` (declining trend)
- `cluster_label == "At Risk / Drop-off"` (clustering result)

## 🤖 LLM Integration (Local Ollama)

The system now uses **Ollama** for local AI analysis. 

### Setup Ollama:
1. Install Ollama from [ollama.com](https://ollama.com)
2. Pull the default model:
   ```bash
   ollama pull llama3.2
   ```
3. Ensure Ollama is running (`ollama serve`).

### Configuration:
You can specify a different model in your `.env` file:
```bash
OLLAMA_MODEL="mistral"
```

**Fallback Logic:**
- IF Ollama is running and returns a valid response → Use local AI insights.
- ELSE → Automatically fallback to the built-in simulated AI engine (which uses real dataset statistics).
- IF AI analysis is disabled/fails → Use Machine Learning pipeline as a secondary fallback.

## 📈 Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| `student_id` | int | Unique student identifier (1-200) |
| `week_number` | int | Week number (1-12) |
| `video_watch_time_minutes` | float | Minutes spent watching videos |
| `quiz_attempts` | int | Number of quiz attempts |
| `quiz_score` | float | Quiz score (0-100) |
| `assignment_submission_count` | int | Assignments submitted |
| `assignment_late_count` | int | Late submissions |
| `coding_assignment_submissions` | int | Coding assignments submitted |
| `coding_success_rate` | float | Success rate (0-1) |
| `discussion_forum_posts` | int | Forum posts count |
| `total_time_spent_minutes` | float | Total time on platform |

## 🛠️ Technology Stack

- **Python 3.9+** – Core language
- **Pandas & NumPy** – Data processing
- **scikit-learn** – Machine learning models
- **Plotly** – Interactive visualizations
- **Streamlit** – Dashboard framework
- **Google Gemini API** – LLM integration
- **Selenium** – Automated notifications

---

<p align="center">
  <em>Built for academic and investor showcase demonstrating AI-driven student engagement analytics.</em>
</p>
