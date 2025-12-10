import os
import mysql.connector

def get_db_connection():
    host = os.environ.get("MYSQLHOST", "localhost")
    user = os.environ.get("MYSQLUSER", "root")
    password = os.environ.get("MYSQLPASSWORD", "")
    database = os.environ.get("MYSQLDATABASE", "tomato_ai")
    port = int(os.environ.get("MYSQLPORT", 3306))

    return mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database,
        port=port
    )
