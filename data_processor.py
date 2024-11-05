# data_processor.py
from pymongo import MongoClient

class DataProcessor:
    def __init__(self, mongo_uri, db_name, collection_name):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.database = None
        self.collection = None

    def connect_to_mongodb(self):
        try:
            # Ensure URI starts with the correct scheme
            if not self.mongo_uri.startswith(('mongodb://', 'mongodb+srv://')):
                raise ValueError("Invalid URI scheme: URI must begin with 'mongodb://' or 'mongodb+srv://'")

            self.client = MongoClient(self.mongo_uri)
            self.database = self.client[self.db_name]
            self.collection = self.database[self.collection_name]
        except Exception as e:
            print(f"An error occurred while connecting to MongoDB: {e}")
            raise

    def fetch_data(self):
        if self.collection is None:
            self.connect_to_mongodb()
        try:
            data_cursor = self.collection.find({}, {'_id': 0})
            datas = list(data_cursor)
            return datas if datas else []
        except Exception as e:
            print(f"An error occurred while fetching data from MongoDB: {e}")
            return []

    def close_connection(self):
        if self.client:
            self.client.close()
