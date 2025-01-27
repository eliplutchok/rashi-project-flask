import os
import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize connection pool as None
connection_pool = None

def initialize_connection_pool():
    global connection_pool
    if connection_pool is None or connection_pool.closed:
        connection_pool = psycopg2.pool.SimpleConnectionPool(
            1, 10,  # Min and max number of connections
            user=os.getenv('DB_USER'),
            host=os.getenv('DB_HOST'),
            database=os.getenv('DB_DATABASE'),
            password=os.getenv('DB_PASSWORD'),
            port=os.getenv('DB_PORT'),
            sslmode='require'  # enables SSL
        )

def get_connection():
    initialize_connection_pool()  # Ensure the pool is initialized before getting a connection
    return connection_pool.getconn()

def release_connection(conn):
    if connection_pool:
        connection_pool.putconn(conn)

def close_pool():
    if connection_pool:
        connection_pool.closeall()