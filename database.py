import psycopg2

def get_connection():
    return psycopg2.connect(
        database="employees",
        user="normal_worker",
        password="apideo",
        host="localhost",
        port=5432
    )