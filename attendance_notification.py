import smtplib
import random
import threading
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from twilio.rest import Client
from datetime import datetime, timedelta
import mysql.connector
import calendar

# Twilio Accounts List (Rotating between them)
TWILIO_ACCOUNTS = [
    {"sid": "AC76cd822028d95980c1a2b82d66569e33", "auth_token": "1999a4de69a03c5c7efe9043e3467eb5", "phone_number": "+17155046578"},
]

# Email Configuration
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
SENDER_EMAIL = 'laantekaisynso@gmail.com'
SENDER_PASSWORD = 'bufi kmzn sfvv nkwn'

# Local DB Config
LOCAL_DB_CONFIG = {
    "host": "192.168.0.113",
    "user": "test",
    "password": "Test@1234",
    "database": "attendance_db"
}

def get_local_connection():
    try:
        connection = mysql.connector.connect(**LOCAL_DB_CONFIG)
        if connection.is_connected():
            return connection
    except mysql.connector.Error as e:
        print(f"❌ MySQL connection error: {e}")
    return None

def get_random_twilio_account():
    return random.choice(TWILIO_ACCOUNTS)

def get_employee_phone(emp_id):
    connection = get_local_connection()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT phone FROM employees WHERE emp_id = %s", (emp_id,))
            result = cursor.fetchone()
            return result[0] if result else None
        except Exception as e:
            print(f"❌ Error fetching phone: {e}")
        finally:
            cursor.close()
            connection.close()
    return None

def send_sms_to_employee(emp_id, employee_name):
    phone_number = get_employee_phone(emp_id)
    phone_number = "+91"+phone_number
    if not phone_number:
        print(f"⚠️ Phone number not found for employee {emp_id}")
        return

    for _ in range(3):
        twilio_account = get_random_twilio_account()
        try:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            client = Client(twilio_account["sid"], twilio_account["auth_token"])
            message = client.messages.create(
                body=f"Hi {employee_name}, your attendance was marked at {now}.",
                from_=twilio_account["phone_number"],
                to=phone_number
            )
            print(f"✅ SMS sent to {employee_name} ({phone_number}) using {twilio_account['phone_number']}")
            return
        except Exception as e:
            print(f"❌ Twilio error, retrying: {e}")
    print(f"🚫 Failed to send SMS to {employee_name}.")

# Get all admin emails
def get_admin_emails():
    connection = get_local_connection()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT email FROM admin")
            results = cursor.fetchall()
            return [email[0] for email in results] if results else []
        except Exception as e:
            print(f"❌ Error fetching admin emails: {e}")
        finally:
            cursor.close()
            connection.close()
    return []

# Send summary email to all admins

def send_daily_summary_email():
    connection = get_local_connection()
    if connection:
        try:
            cursor = connection.cursor()
            today = datetime.now()
            today_str = today.strftime('%Y-%m-%d')

            # Fetch today's attendance
            cursor.execute("SELECT emp_id, name, timestamp FROM attendance WHERE DATE(timestamp) = %s", (today_str,))
            records = cursor.fetchall()

            if not records:
                print("ℹ️ No attendance records for today.")
                return

            # Get all employees
            cursor.execute("""
                SELECT e.emp_id, e.name 
                FROM employees e
                JOIN status s ON e.emp_id = s.emp_id
                WHERE s.active_status = 'active'
            """)
            all_employees = cursor.fetchall()
            emp_info = {emp_id: name for emp_id, name in all_employees}

            # --- WEEKLY ABSENTEE ANALYSIS (Mon to Today) ---
            start_of_week = today - timedelta(days=today.weekday())  # Monday
            end_of_week = today  # Only up to today

            # Generate weekdays between Monday and today
            working_days_week = [
                (start_of_week + timedelta(days=i)).date()
                for i in range((end_of_week - start_of_week).days + 1)
                if (start_of_week + timedelta(days=i)).weekday() < 5
            ]

            cursor.execute("""
                SELECT emp_id, DATE(timestamp) as date
                FROM attendance
                WHERE DATE(timestamp) BETWEEN %s AND %s
            """, (start_of_week.date(), end_of_week.date()))
            week_records = cursor.fetchall()

            week_attendance = {}
            for emp_id, date in week_records:
                week_attendance.setdefault(emp_id, set()).add(date)

            weekly_absentees = []
            for emp_id in emp_info:
                attended_days = week_attendance.get(emp_id, set())
                absents = sum(1 for day in working_days_week if day not in attended_days)
                if absents > 2:
                    weekly_absentees.append(f"- ID: {emp_id}, Name: {emp_info[emp_id]}, Missed Days: {absents} (Week-to-date)")

            # --- MONTHLY ABSENTEE ANALYSIS (1st to Today) ---
            start_of_month = today.replace(day=1)
            end_of_month = today

            # Only include weekdays up to today
            working_days_month = [
                (start_of_month + timedelta(days=i)).date()
                for i in range((end_of_month - start_of_month).days + 1)
                if (start_of_month + timedelta(days=i)).weekday() < 5
            ]

            cursor.execute("""
                SELECT emp_id, DATE(timestamp) as date
                FROM attendance
                WHERE DATE(timestamp) BETWEEN %s AND %s
            """, (start_of_month, end_of_month))
            month_records = cursor.fetchall()

            month_attendance = {}
            for emp_id, date in month_records:
                month_attendance.setdefault(emp_id, set()).add(date)

            monthly_absentees = []
            for emp_id in emp_info:
                attended_days = month_attendance.get(emp_id, set())
                absents = sum(1 for day in working_days_month if day not in attended_days)
                if absents > 5:
                    monthly_absentees.append(f"- ID: {emp_id}, Name: {emp_info[emp_id]}, Missed Days: {absents} (Month-to-date)")

            # Build today's attendance lines
            attendance_lines = '\n'.join(
                [f"- ID: {emp_id}, Name: {name}, Time: {ts.strftime('%H:%M:%S')}" for emp_id, name, ts in records]
            )

            weekly_absent_lines = "\n".join(weekly_absentees) if weekly_absentees else "None"
            monthly_absent_lines = "\n".join(monthly_absentees) if monthly_absentees else "None"

            admin_emails = get_admin_emails()
            if not admin_emails:
                print("No admin emails found in database.")
                return

            body = (
                f"Dear Admin,\n\n"
                f"📅 Attendance Summary for {today_str}:\n"
                f"{attendance_lines}\n\n"
                f"Employees Absent More Than 2 Days This Week (Week-to-date):\n{weekly_absent_lines}\n\n"
                f"Employees Absent More Than 5 Days This Month (Month-to-date, excluding weekends):\n{monthly_absent_lines}\n\n"
                f"Best regards,\n"
                f"Your Automated Attendance System"
            )

            msg = MIMEMultipart()
            msg['From'] = f"Attendance System <{SENDER_EMAIL}>"
            msg['To'] = ", ".join(admin_emails)
            msg['Subject'] = f"Daily Attendance Summary - {today_str}"
            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
            server.quit()

            print(f"✅ Summary email sent to admins: {', '.join(admin_emails)}")
        except Exception as e:
            print(f"❌ Error sending summary email: {e}")
        finally:
            cursor.close()
            connection.close()

def schedule_daily_email(hour, minute):
    def run_scheduler():
        while True:
            now = datetime.now()
            target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if now > target_time:
                target_time += timedelta(days=1)
            wait_time = (target_time - now).total_seconds()
            print(f"⏳ Waiting {wait_time / 60:.2f} minutes to send email...")
            time.sleep(wait_time)
            send_daily_summary_email()

    threading.Thread(target=run_scheduler, daemon=True).start()

def get_admin_phones():
    connection = get_local_connection()
    if connection:
        try:
            cursor = connection.cursor()
            query = """
                SELECT DISTINCT e.phone
                FROM admin a
                INNER JOIN employees e ON a.email = e.email
                WHERE e.phone IS NOT NULL
            """
            cursor.execute(query)
            results = cursor.fetchall()
            return [f"+91{row[0]}" for row in results if row[0]]  # assuming Indian numbers
        except Exception as e:
            print(f"❌ Error fetching admin phone numbers: {e}")
        finally:
            cursor.close()
            connection.close()
    return []

def send_admin_alert_sms(emp_id, employee_name):
    now = datetime.now()

    # ✅ FIXED TIME WINDOW: 11:45 AM to 11:50 AM
    if now.hour == 20 and now.hour == 8:
        twilio_account = get_random_twilio_account()

        # Check if the employee is known or unknown
        if emp_id and employee_name:
            alert_msg = f"⚠️ Security Alert: {employee_name} (ID: {emp_id}) entered the premises at {now.strftime('%H:%M:%S')}."
        else:
            alert_msg = f"⚠️ Security Alert: Unknown person entered the premises at {now.strftime('%H:%M:%S')}."

        admin_phones = get_admin_phones()
        print(f"📞 Admin phones: {admin_phones}")  # ✅ Debug log

        for phone in admin_phones:
            try:
                client = Client(twilio_account["sid"], twilio_account["auth_token"])
                client.messages.create(
                    body=alert_msg,
                    from_=twilio_account["phone_number"],
                    to=phone
                )
                print(f"📨 Alert SMS sent to admin: {phone}")
            except Exception as e:
                print(f"❌ Failed to send admin alert SMS to {phone}: {e}")
