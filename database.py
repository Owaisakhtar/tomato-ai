import os
import mysql.connector

def get_db_connection():
    host = os.environ.get("MYSQLHOST", "mysql.railway.internal")
    user = os.environ.get("MYSQLUSER", "root")
    password = os.environ.get("MYSQLPASSWORD", "")
    database = os.environ.get("MYSQLDATABASE", "railway")
    port = int(os.environ.get("MYSQLPORT", 3306))

    try:
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=port
        )
        print("✅ DB connection successful")
        return conn
    except mysql.connector.Error as e:
        print("❌ DB connection error:", e)
        raise
