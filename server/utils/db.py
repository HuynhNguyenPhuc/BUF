import pymongo
from dotenv import load_dotenv
import os

load_dotenv()

class MongoDBClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.client = pymongo.MongoClient(os.getenv("MONGODB_URI"))
        return cls._instance

    def client(self):
        return self._instance.client

    def getDatabase(self, databaseName):
        return self.client[databaseName]
