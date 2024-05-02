from kafka import KafkaConsumer
from dotenv import load_dotenv
import os, sys, json
from s3fs import S3FileSystem

sys.path.append("C:/Users/User/Desktop/Project/Stock Prediction/server")

from utils.db import MongoDBClient

load_dotenv()

consumer = KafkaConsumer(
    'UPCOM',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

storage = "MongoDB"  # "S3" or "MongoDB"

if storage == "S3":
    print("Use S3 clod storage")
    s3 = S3FileSystem(
        key=os.getenv("ACCESS_KEY"),
        secret=os.getenv("SECRET_ACCESS_KEY")
    )

    for count, i in enumerate(consumer):
        with s3.open("s3://vietnamese-stock-exchange/UPCOM/stock_data_{}.json".format(count), 'w') as file:
            json.dump(i.value, file)

elif storage == "MongoDB":
    print("Use MongoDB storage")
    client = MongoDBClient()
    db = client.getDatabase("exchange")
    collection = db["UPCOM"]

    for count, i in enumerate(consumer):
        document = i.value
        collection.insert_one(document)