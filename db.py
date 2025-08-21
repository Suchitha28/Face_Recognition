import mysql.connector
from mysql.connector import Error
from datetime import datetime

# MySQL Database Configuration (Local Database)
LOCAL_DB_CONFIG = {
    "host": "192.168.0.113",
    "user": "suchita",
    "password":"suchita@1234",
    "database": "attendance_db"
}

# Function to Connect to the Local Database
def get_local_connection():
    try:
        connection = mysql.connector.connect(**LOCAL_DB_CONFIG)
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
    return None

# Function to Check if Attendance Exists for Today
def attendance_exists(emp_id):
    connection = get_local_connection()
    if connection:
        try:
            cursor = connection.cursor()
            today_date = datetime.now().date()
            query = """
            SELECT COUNT(*) FROM attendance 
            WHERE emp_id = %s AND DATE(timestamp) = %s
            """
            cursor.execute(query, (emp_id, today_date))
            count = cursor.fetchone()[0]
            return count > 0
        except Error as e:
            print(f"Error checking attendance: {e}")
            return False
        finally:
            cursor.close()
            connection.close()
    return False

def is_employee_active(emp_id):
    connection = get_local_connection()
    if connection:
        try:
            cursor = connection.cursor()
            query = "SELECT active_status FROM status WHERE emp_id = %s"
            cursor.execute(query, (emp_id,))
            result = cursor.fetchone()
            if result:
                return result[0].lower() == "active"
            return False  # Not found treated as inactive
        except Error as e:
            print(f"Error checking employee status: {e}")
            return False
        finally:
            cursor.close()
            connection.close()
    return False


# Modified Function to Insert Attendance
def insert_record(emp_id, name, timestamp):
    if not is_employee_active(emp_id):
        print(f"Employee {emp_id} is not active. Attendance not recorded.")
        return

    if attendance_exists(emp_id):
        print(f"Attendance for employee {emp_id} already recorded today.")
        return

    connection = get_local_connection()
    if connection:
        try:
            cursor = connection.cursor()
            query = """
            INSERT INTO attendance (emp_id, name, timestamp) 
            VALUES (%s, %s, %s)
            """
            cursor.execute(query, (emp_id, name, timestamp))
            connection.commit()
            print("Record inserted into local database successfully.")
        except Error as e:
            print(f"Error inserting record: {e}")
        finally:
            cursor.close()
            connection.close()
    else:
        print("Failed to connect to local database.")

# Function to Create Attendance and Status Tables if Not Exist
def create_tables():
    connection = get_local_connection()
    if connection:
        try:
            cursor = connection.cursor()

            # Create attendance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    emp_id VARCHAR(50),
                    name VARCHAR(100),
                    timestamp DATETIME
                )
            """)

            # ✅ Create status table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS status (
                    emp_id VARCHAR(50) PRIMARY KEY,
                    name VARCHAR(100),
                    active_status VARCHAR(20)  -- should be 'active' or 'inactive'
                )
            """)

            connection.commit()
            print("Attendance and Status tables ensured in local database.")
        except Error as e:
            print(f"Error creating tables: {e}")
        finally:
            cursor.close()
            connection.close()

# Ensure Tables Exist on Startup
if __name__ == "__main__":
    create_tables()