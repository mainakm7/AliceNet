from pymongo import MongoClient
import threading

MONGO_URI = "mongodb://localhost:8081"
DB_NAME = "my_database"

class Database:
    client = None
    db = None
    _instance_lock = threading.Lock()
    _unique_instance = None
    
    def __new__(cls):
        # Ensure thread-safe singleton instantiation
        with cls._instance_lock:
            if cls._unique_instance is None:
                cls._unique_instance = super(Database, cls).__new__(cls)
        return cls._unique_instance

    @classmethod
    def initialize(cls):
        if cls.client is None:
            cls.client = MongoClient(MONGO_URI)
            cls.db = cls.client[DB_NAME]

    @classmethod
    def get_db(cls):
        if cls.db is None:
            cls.initialize()  # Ensure the database is initialized
        return cls.db

    @classmethod
    def close(cls):
        if cls.client:
            cls.client.close()
            cls.client = None
            cls.db = None
