"""
EduPulse – Teacher-Student Mapping Generator
Creates a realistic mapping of teachers to students for the LMS system.
Each teacher is assigned a group of students they are responsible for.
"""

import random
import json
import os

# ─── Configuration ───────────────────────────────────────────────────────────
NUM_TEACHERS = 10
NUM_STUDENTS = 200
SEED = 42

# Teacher profiles with department info
TEACHER_PROFILES = [
    {"teacher_id": "TCH001", "name": "Dr. Ananya Sharma",    "department": "Computer Science",  "subject": "Data Structures & Algorithms"},
    {"teacher_id": "TCH002", "name": "Prof. Rajesh Kumar",   "department": "Computer Science",  "subject": "Operating Systems"},
    {"teacher_id": "TCH003", "name": "Dr. Priya Menon",      "department": "Information Technology", "subject": "Database Management"},
    {"teacher_id": "TCH004", "name": "Prof. Vikram Singh",   "department": "Computer Science",  "subject": "Machine Learning"},
    {"teacher_id": "TCH005", "name": "Dr. Kavitha Rajan",    "department": "Electronics",       "subject": "Digital Signal Processing"},
    {"teacher_id": "TCH006", "name": "Prof. Arjun Nair",     "department": "Information Technology", "subject": "Web Technologies"},
    {"teacher_id": "TCH007", "name": "Dr. Meera Patel",      "department": "Mathematics",       "subject": "Linear Algebra"},
    {"teacher_id": "TCH008", "name": "Prof. Suresh Iyer",    "department": "Computer Science",  "subject": "Computer Networks"},
    {"teacher_id": "TCH009", "name": "Dr. Lakshmi Devi",     "department": "Information Technology", "subject": "Cloud Computing"},
    {"teacher_id": "TCH010", "name": "Prof. Deepak Reddy",   "department": "Computer Science",  "subject": "Software Engineering"},
]


def generate_mapping() -> dict:
    """
    Generate teacher-to-student mapping.
    
    Each teacher is assigned ~20 students (200 students / 10 teachers).
    Returns a dict keyed by teacher_id.
    """
    random.seed(SEED)
    
    # Shuffle student IDs and distribute evenly
    student_ids = list(range(1, NUM_STUDENTS + 1))
    random.shuffle(student_ids)
    
    students_per_teacher = NUM_STUDENTS // NUM_TEACHERS
    
    mapping = {}
    for i, teacher in enumerate(TEACHER_PROFILES):
        start_idx = i * students_per_teacher
        end_idx = start_idx + students_per_teacher
        assigned_students = sorted(student_ids[start_idx:end_idx])
        
        mapping[teacher["teacher_id"]] = {
            **teacher,
            "student_ids": assigned_students,
            "student_count": len(assigned_students),
        }
    
    return mapping


def save_mapping(mapping: dict, output_path: str = None):
    """Save the mapping to a JSON file."""
    if output_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(base_dir, "data", "teacher_student_mapping.json")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"✅ Teacher-Student mapping saved to: {output_path}")
    return output_path


def load_mapping(filepath: str = None) -> dict:
    """Load the mapping from JSON file, or generate if not exists."""
    if filepath is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, "data", "teacher_student_mapping.json")
    
    if not os.path.exists(filepath):
        print("⚠️  Mapping file not found. Generating new mapping...")
        mapping = generate_mapping()
        save_mapping(mapping, filepath)
        return mapping
    
    with open(filepath, "r") as f:
        return json.load(f)


def get_teacher_info(teacher_id: str, mapping: dict) -> dict:
    """Get teacher info and their assigned students."""
    teacher_id = teacher_id.upper()
    if teacher_id in mapping:
        return mapping[teacher_id]
    return None


def list_all_teachers(mapping: dict):
    """Print a formatted list of all teachers."""
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                    📚 REGISTERED TEACHERS                      │")
    print("├───────────┬──────────────────────────┬──────────────┬──────────┤")
    print("│ Teacher ID│ Name                     │ Department   │ Students │")
    print("├───────────┼──────────────────────────┼──────────────┼──────────┤")
    for tid, info in mapping.items():
        name = info['name'][:24].ljust(24)
        dept = info['department'][:12].ljust(12)
        count = str(info['student_count']).center(8)
        print(f"│ {tid}   │ {name} │ {dept} │ {count} │")
    print("└───────────┴──────────────────────────┴──────────────┴──────────┘")


# ─── Standalone execution ────────────────────────────────────────────────────
if __name__ == "__main__":
    mapping = generate_mapping()
    save_mapping(mapping)
    list_all_teachers(mapping)
    
    # Show a sample
    sample_teacher = "TCH001"
    info = get_teacher_info(sample_teacher, mapping)
    if info:
        print(f"\n📋 Sample - {info['name']} ({sample_teacher}):")
        print(f"   Department: {info['department']}")
        print(f"   Subject: {info['subject']}")
        print(f"   Students: {info['student_ids']}")
