"""
EduPulse – Selenium Alert Bot
Automates sending teacher notifications through a simulated web interface.
Uses Selenium to demonstrate automated alert delivery.
"""

import os
import time
import json
import pandas as pd  # type: ignore
from datetime import datetime

try:
    from selenium import webdriver  # type: ignore
    from selenium.webdriver.common.by import By  # type: ignore
    from selenium.webdriver.common.keys import Keys  # type: ignore
    from selenium.webdriver.chrome.service import Service  # type: ignore
    from selenium.webdriver.chrome.options import Options  # type: ignore
    from selenium.webdriver.support.ui import WebDriverWait  # type: ignore
    from selenium.webdriver.support import expected_conditions as EC  # type: ignore
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False


# ─── Configuration ───────────────────────────────────────────────────────────
ALERT_LOG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "alert_log.json",
)

NOTIFICATION_HTML_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "automation",
    "notification_page.html",
)


def create_notification_page():
    """Create a local HTML page that simulates a teacher notification portal."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EduPulse - Teacher Notification Portal</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 40px;
        }
        .container { max-width: 800px; margin: 0 auto; }
        h1 {
            text-align: center;
            font-size: 2rem;
            background: linear-gradient(90deg, #ff6b6b, #ffa500, #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #b0b0b0;
        }
        input, textarea {
            width: 100%;
            padding: 12px 16px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            color: #fff;
            font-size: 14px;
        }
        textarea { min-height: 120px; resize: vertical; }
        input:focus, textarea:focus {
            outline: none;
            border-color: #ff6b6b;
            box-shadow: 0 0 10px rgba(255,107,107,0.3);
        }
        .btn-send {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #ff6b6b, #ee5a24);
            border: none;
            border-radius: 8px;
            color: white;
            font-size: 16px;
            font-weight: 700;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .btn-send:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(255,107,107,0.3);
        }
        #notification-log {
            margin-top: 30px;
            padding: 20px;
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.05);
        }
        .log-entry {
            padding: 12px;
            margin-bottom: 10px;
            background: rgba(255,107,107,0.1);
            border-left: 3px solid #ff6b6b;
            border-radius: 4px;
            font-size: 13px;
        }
        .success {
            background: rgba(46,213,115,0.1);
            border-left-color: #2ed573;
            color: #2ed573;
        }
        #status { 
            text-align: center; 
            margin-top: 15px; 
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔔 EduPulse Teacher Alert Portal</h1>
        
        <div class="form-group">
            <label for="teacher-email">Teacher Email</label>
            <input type="email" id="teacher-email" placeholder="teacher@university.edu">
        </div>
        
        <div class="form-group">
            <label for="student-id">Student ID</label>
            <input type="text" id="student-id" placeholder="e.g., 104">
        </div>
        
        <div class="form-group">
            <label for="alert-message">Alert Message</label>
            <textarea id="alert-message" placeholder="Enter the alert details..."></textarea>
        </div>
        
        <button class="btn-send" id="send-btn" onclick="sendNotification()">
            🚀 Send Alert Notification
        </button>
        
        <div id="status"></div>
        
        <div id="notification-log">
            <h3 style="margin-bottom: 15px; color: #b0b0b0;">📋 Notification Log</h3>
            <div id="log-entries"></div>
        </div>
    </div>

    <script>
        let notificationCount = 0;
        
        function sendNotification() {
            const email = document.getElementById('teacher-email').value;
            const studentId = document.getElementById('student-id').value;
            const message = document.getElementById('alert-message').value;
            
            if (!email || !studentId || !message) {
                document.getElementById('status').innerHTML = 
                    '<span style="color: #ff6b6b;">⚠️ Please fill all fields</span>';
                return;
            }
            
            notificationCount++;
            const timestamp = new Date().toLocaleTimeString();
            
            const logEntry = document.createElement('div');
            logEntry.className = 'log-entry success';
            logEntry.innerHTML = `✅ Alert #${notificationCount} sent at ${timestamp}<br>
                To: ${email} | Student: ${studentId}<br>
                ${message.substring(0, 100)}...`;
            
            document.getElementById('log-entries').prepend(logEntry);
            document.getElementById('status').innerHTML = 
                `<span style="color: #2ed573;">✅ Notification sent successfully! (${notificationCount} total)</span>`;
            
            // Clear form
            document.getElementById('alert-message').value = '';
            document.getElementById('student-id').value = '';
        }
    </script>
</body>
</html>"""

    os.makedirs(os.path.dirname(NOTIFICATION_HTML_PATH), exist_ok=True)
    with open(NOTIFICATION_HTML_PATH, "w") as f:
        f.write(html_content)
    print(f"✅ Notification page created: {NOTIFICATION_HTML_PATH}")
    return NOTIFICATION_HTML_PATH


def send_alerts_via_selenium(alerts_df: pd.DataFrame, max_alerts: int = 5) -> list:
    """
    Use Selenium to automatically send teacher alert notifications
    through the simulated web portal.
    
    Args:
        alerts_df: DataFrame of alerts to send
        max_alerts: Maximum number of alerts to send in this batch
    
    Returns:
        List of sent alert records
    """
    sent_alerts = []

    if not SELENIUM_AVAILABLE:
        print("⚠️  Selenium not installed. Using log-based alert delivery.")
        return _fallback_log_alerts(alerts_df, max_alerts)

    # Create the notification page
    page_path = create_notification_page()
    page_url = f"file://{page_path}"

    try:
        # Configure Chrome
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1200,800")

        driver = webdriver.Chrome(options=chrome_options)
        driver.get(page_url)
        time.sleep(1)

        # Send alerts for critical students
        critical_alerts = alerts_df[alerts_df["severity"] == "critical"].head(max_alerts)

        for _, alert in critical_alerts.iterrows():
            try:
                # Fill in the form
                email_field = driver.find_element(By.ID, "teacher-email")
                email_field.clear()
                email_field.send_keys("teacher@edupulse.edu")

                student_field = driver.find_element(By.ID, "student-id")
                student_field.clear()
                student_field.send_keys(str(alert["student_id"]))

                message_field = driver.find_element(By.ID, "alert-message")
                message_field.clear()
                message_field.send_keys(alert["alert_message"])

                # Click send
                send_btn = driver.find_element(By.ID, "send-btn")
                send_btn.click()
                time.sleep(0.5)

                sent_alerts.append({
                    "student_id": int(alert["student_id"]),
                    "message": alert["alert_message"],
                    "sent_via": "selenium",
                    "timestamp": datetime.now().isoformat(),
                    "status": "sent",
                })

                print(f"   📨 Alert sent for Student {alert['student_id']}")

            except Exception as e:
                print(f"   ❌ Failed to send alert for Student {alert['student_id']}: {e}")
                sent_alerts.append({
                    "student_id": int(alert["student_id"]),
                    "message": alert["alert_message"],
                    "sent_via": "selenium",
                    "timestamp": datetime.now().isoformat(),
                    "status": f"failed: {str(e)}",
                })

        driver.quit()
        print(f"\n✅ Selenium: {len(sent_alerts)} alerts processed")

    except Exception as e:
        print(f"⚠️  Selenium driver failed: {e}")
        print("   Falling back to log-based delivery...")
        return _fallback_log_alerts(alerts_df, max_alerts)

    # Save log
    _save_alert_log(sent_alerts)
    return sent_alerts


def _fallback_log_alerts(alerts_df: pd.DataFrame, max_alerts: int = 5) -> list:
    """Fallback: Log alerts to file when Selenium is unavailable."""
    sent_alerts = []
    critical_alerts = alerts_df[alerts_df["severity"] == "critical"].head(max_alerts)

    for _, alert in critical_alerts.iterrows():
        record = {
            "student_id": int(alert["student_id"]),
            "message": alert["alert_message"],
            "sent_via": "log_file",
            "timestamp": datetime.now().isoformat(),
            "status": "logged",
        }
        sent_alerts.append(record)
        print(f"   📝 Alert logged for Student {alert['student_id']}")

    _save_alert_log(sent_alerts)
    print(f"\n✅ Fallback: {len(sent_alerts)} alerts logged to file")
    return sent_alerts


def _save_alert_log(alerts: list):
    """Save alert log to JSON file."""
    os.makedirs(os.path.dirname(ALERT_LOG_FILE), exist_ok=True)

    existing = []
    if os.path.exists(ALERT_LOG_FILE):
        with open(ALERT_LOG_FILE, "r") as f:
            existing = json.load(f)

    existing.extend(alerts)

    with open(ALERT_LOG_FILE, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"   📄 Alert log saved: {ALERT_LOG_FILE}")


# ─── Standalone execution ────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))
    from dataset_generator import generate_dataset  # type: ignore
    from feature_engineering import engineer_features, get_student_summary  # type: ignore
    from ml_models import run_ml_pipeline  # type: ignore
    from alert_system import generate_alerts  # type: ignore

    df = generate_dataset()
    df = engineer_features(df)
    summary = get_student_summary(df)
    summary, _, _, _ = run_ml_pipeline(summary)
    alerts = generate_alerts(summary)

    print("\n🤖 Starting Selenium Alert Bot...")
    sent = send_alerts_via_selenium(alerts)
    print(f"\n✅ Total alerts sent: {len(sent)}")
