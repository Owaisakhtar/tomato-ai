# database.py
import os
import mysql.connector

def get_db_connection():
    host = os.environ.get("DB_HOST", "localhost")
    user = os.environ.get("DB_USER", "root")
    password = os.environ.get("DB_PASS", "")
    database = os.environ.get("DB_NAME", "tomato_ai")
    # optional: port
    port = int(os.environ.get("DB_PORT", 3306))

    return mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database,
        port=port
    )
