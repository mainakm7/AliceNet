from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:8081"
DB_NAME = "my_database"

class Database:
    client = None
    db = None

    @classmethod
    def initialize(cls):
        cls.client = MongoClient(MONGO_URI)
        cls.db = cls.client[DB_NAME]

    @classmethod
    def get_db(cls):
        return cls.db

    @classmethod
    def close(cls):
        if cls.client:
            cls.client.close()
