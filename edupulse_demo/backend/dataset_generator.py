"""
EduPulse – Dataset Generator
Generates synthetic student LMS activity logs for 200 students
with realistic engagement patterns across multiple weeks.
"""

import os
import numpy as np
import pandas as pd

# ─── Configuration ───────────────────────────────────────────────────────────
NUM_STUDENTS = 200
NUM_WEEKS = 12
SEED = 42

# Student engagement category distribution
CATEGORY_DISTRIBUTION = {
    "high_engagement": 0.30,      # 30% highly engaged
    "moderate_engagement": 0.35,  # 35% moderately engaged
    "low_engagement": 0.20,       # 20% low engagement
    "dropoff_behavior": 0.15,     # 15% drop-off pattern
}

np.random.seed(SEED)


def _assign_categories(num_students: int) -> list:
    """Assign engagement categories to students based on distribution."""
    categories = []
    for category, proportion in CATEGORY_DISTRIBUTION.items():
        count = int(num_students * proportion)
        categories.extend([category] * count)
    # Fill any remainder
    while len(categories) < num_students:
        categories.append("moderate_engagement")
    np.random.shuffle(categories)
    return categories


def _generate_student_week(student_id: int, week: int, category: str) -> dict:
    """Generate a single week of activity for a student based on their category."""

    if category == "high_engagement":
        video = np.random.normal(120, 15)
        quiz_attempts = np.random.randint(3, 7)
        quiz_score = np.random.normal(85, 8)
        assignments = np.random.randint(3, 6)
        late = np.random.choice([0, 0, 0, 1], p=[0.7, 0.1, 0.1, 0.1])
        coding_subs = np.random.randint(2, 5)
        coding_rate = np.random.normal(0.85, 0.08)
        forum_posts = np.random.randint(3, 10)
        total_time = np.random.normal(300, 40)

    elif category == "moderate_engagement":
        video = np.random.normal(70, 20)
        quiz_attempts = np.random.randint(1, 4)
        quiz_score = np.random.normal(65, 12)
        assignments = np.random.randint(1, 4)
        late = np.random.choice([0, 1, 2], p=[0.5, 0.35, 0.15])
        coding_subs = np.random.randint(1, 3)
        coding_rate = np.random.normal(0.60, 0.12)
        forum_posts = np.random.randint(0, 5)
        total_time = np.random.normal(180, 50)

    elif category == "low_engagement":
        video = np.random.normal(25, 12)
        quiz_attempts = np.random.randint(0, 2)
        quiz_score = np.random.normal(40, 15)
        assignments = np.random.randint(0, 2)
        late = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.3, 0.2])
        coding_subs = np.random.randint(0, 2)
        coding_rate = np.random.normal(0.35, 0.15)
        forum_posts = np.random.randint(0, 2)
        total_time = np.random.normal(60, 25)

    elif category == "dropoff_behavior":
        # Engagement decays over weeks
        decay = max(0.05, 1.0 - (week / NUM_WEEKS) * 1.2)
        video = np.random.normal(100 * decay, 15)
        quiz_attempts = max(0, int(np.random.normal(4 * decay, 1)))
        quiz_score = np.random.normal(75 * decay, 10)
        assignments = max(0, int(np.random.normal(3 * decay, 1)))
        late = np.random.choice([0, 1, 2, 3], p=[0.3 * decay, 0.3, 0.2, max(0.01, 0.2 + 0.3 * (1 - decay))])
        # Renormalize probabilities
        coding_subs = max(0, int(np.random.normal(3 * decay, 1)))
        coding_rate = np.random.normal(0.7 * decay, 0.12)
        forum_posts = max(0, int(np.random.normal(5 * decay, 2)))
        total_time = np.random.normal(250 * decay, 30)
    else:
        raise ValueError(f"Unknown category: {category}")

    return {
        "student_id": student_id,
        "week_number": week,
        "video_watch_time_minutes": round(max(0, video), 1),
        "quiz_attempts": max(0, int(quiz_attempts)),
        "quiz_score": round(np.clip(quiz_score, 0, 100), 1),
        "assignment_submission_count": max(0, int(assignments)),
        "assignment_late_count": max(0, int(late)),
        "coding_assignment_submissions": max(0, int(coding_subs)),
        "coding_success_rate": round(np.clip(coding_rate, 0, 1), 2),
        "discussion_forum_posts": max(0, int(forum_posts)),
        "total_time_spent_minutes": round(max(0, total_time), 1),
    }


def generate_dataset(output_path: str = None) -> pd.DataFrame:
    """
    Generate complete synthetic student activity dataset.
    
    Args:
        output_path: If provided, saves CSV to this path.
    
    Returns:
        DataFrame with student activity logs.
    """
    categories = _assign_categories(NUM_STUDENTS)
    records = []

    for idx, category in enumerate(categories):
        student_id = idx + 1
        for week in range(1, NUM_WEEKS + 1):
            record = _generate_student_week(student_id, week, category)
            record["engagement_category"] = category
            records.append(record)

    df = pd.DataFrame(records)

    # Sort for readability
    df = df.sort_values(["student_id", "week_number"]).reset_index(drop=True)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✅ Dataset generated: {len(df)} rows ({NUM_STUDENTS} students × {NUM_WEEKS} weeks)")
        print(f"   Saved to: {output_path}")

    return df


# ─── Standalone execution ────────────────────────────────────────────────────
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output = os.path.join(base_dir, "data", "student_activity_logs.csv")
    df = generate_dataset(output)

    # Print summary
    print("\n📊 Dataset Summary:")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n   Category Distribution:")
    for cat, count in df.groupby("engagement_category")["student_id"].nunique().items():
        print(f"     • {cat}: {count} students")
