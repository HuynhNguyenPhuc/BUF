import pandas as pd
from kafka import KafkaProducer
from time import sleep
import json
import os

producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer= lambda x: json.dumps(x).encode('utf-8'))
directory = "data/exchanges"
for exchange in os.listdir(directory):
    topic = exchange
    exchange_dir = os.path.join(directory, exchange)
    stocks = os.listdir(exchange_dir)[:5]
    for stock in stocks:
        stock_name = os.path.splitext(stock)[0]
        df = pd.read_csv(os.path.join(exchange_dir, stock))
        df = df[["Date", "Open", "Close", "High", "Low", "Volume"]]
        records = df.to_dict(orient="records")
        for record in records:
            record["Stock"] = stock_name
            record["Exchange"] = exchange
            producer.send(topic, value = record)
            sleep(0.01)

