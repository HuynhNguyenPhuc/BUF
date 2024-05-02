import numpy as np
import pandas as pd
import tensorflow as tf

from fastapi import HTTPException

import datetime
from utils.datetime import is_weekend

from utils.generator import WindowGenerator
from utils.getter import getPretrainedModel
from utils.loss import rmse
from model.model import LSTM_Model
from keras.callbacks import ModelCheckpoint
from utils.loss import mse, rmse, mape, r2_score

from utils.db import MongoDBClient

INPUT_SIZE = 50
NUM_FEATURES = 4
LABEL_SIZE = 1
UNITS = 64

class Controller():
    def __init__(self):
        self.client = MongoDBClient()
        self.stockExchanges = self.getExchangeList()
        self.model = getPretrainedModel()
        self.generator = WindowGenerator(input_size=INPUT_SIZE, label_size=LABEL_SIZE)
        
    def getExchangeList(self):
        database = self.client.getDatabase("exchange")
        return database.list_collection_names()
    
    def getStockList(self, exchange):
        database = self.client.getDatabase("exchange")
        collection = database.get_collection(exchange)
        return collection.distinct("Stock")

    def loadAll(self):
        input_data = {}
        label_data = {}

        database = self.client.getDatabase("exchange")

        for exchange in self.stockExchanges:
            input_data[exchange] = []
            label_data[exchange] = []

            collection = database.get_collection(exchange)
            documents = collection.find()

            data_list = [document for document in documents]

            df = pd.DataFrame(data_list)
            df['Date'] = pd.to_datetime(df['Date'], format="%m/%d/%Y")
            df.sort_values(by='Date', inplace=True)

            for stock, group_df in df.groupby('Stock'):
                stock_input, stock_label = self.generator.generateWindow(group_df[['Open', 'Close', 'High', 'Low']], n_features=4)
                input_data[exchange].append(stock_input)
                label_data[exchange].append(stock_label)

            input_data[exchange] = np.concatenate(input_data[exchange])
            label_data[exchange] = np.concatenate(label_data[exchange])

        return input_data, label_data

    def load(self, stock_exchange, stock_name):
        database = self.client.getDatabase("exchange")
        collection = database.get_collection(stock_exchange)
        documents = collection.find({"Stock": stock_name})

        data = [document for document in documents]

        if data:
            df = pd.DataFrame(data)
            df['Date'] = pd.to_datetime(df['Date'], format="%m/%d/%Y")
            df.sort_values(by='Date', inplace=True)

            documents = df.to_dict(orient="records")

            return documents
        return None
    
    def train(self, inputSize, numFeatures, units, optimizer, epochs, batchSize, validationSplit):
        input_data, label_data = self.loadAll()
        for i in self.stockExchanges:
            self.model[i] = LSTM_Model(inputSize, numFeatures, units)
            self.model[i].compile(optimizer, loss=rmse)

        with tf.device('/GPU:0'):
            for i in self.stockExchanges:
                checkpoint_path = f"model/results/{i}_model.h5"
                checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path,
                                            save_weights_only=True,
                                            save_best_only=True,
                                            save_freq='epoch')
                self.model[i].fit(input_data[i], label_data[i],
                                        epochs = epochs,
                                        batch_size = batchSize,
                                        verbose = 1,
                                        validation_split = validationSplit,
                                        shuffle = True,
                                        callbacks=[checkpoint_callback])

    def predict(self, stock_name, stock_exchange, date):
        documents = self.load(stock_exchange, stock_name)

        if documents is None:
            return None

        predictor = self.model[stock_exchange]

        input_data = np.array([[doc['Open'], doc['High'], doc['Low'], doc['Close']] for doc in documents])
        input_size, num_features = input_data.shape
        input_data = input_data.reshape(1, input_size, num_features)[:,-50:,:]

        current_date = documents[-1]['Date'] + datetime.timedelta(days=1)
        predicted_data = None

        while current_date.strftime("%d/%m/%Y") != date:
            if is_weekend(current_date):
                current_date += datetime.timedelta(days=1)
                continue
            predicted_data = predictor.predict(input_data, verbose=0)
            input_data = np.concatenate((input_data[:, 1:, :], predicted_data.reshape(1, 1, num_features)), axis=1)
            current_date += datetime.timedelta(days=1)

        if predicted_data is None:
            raise HTTPException(status_code=400, detail="Invalid date!")

        return predicted_data.tolist()
    
    def evaluation(self):
        input_data, label_data = self.loadAll()
        model = getPretrainedModel()

        mse_ = []
        rmse_ = []
        mape_ = []
        r2_score_ = []

        for exchange in self.stockExchanges:
            y_true = label_data[exchange]
            y_pred = model[exchange].predict(input_data[exchange], verbose = 0)

            mse_.append(mse(y_true, y_pred).numpy())
            rmse_.append(rmse(y_true, y_pred).numpy())
            mape_.append(mape(y_true, y_pred).numpy())
            r2_score_.append(r2_score(y_true, y_pred).numpy())

        results = [{
            'Exchange': exchange,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R2 Score': r2_score
        } for exchange, mse, rmse, mape, r2_score in zip(self.stockExchanges, mse_, rmse_, mape_, r2_score_)]
        results_df = pd.DataFrame(results)
        results_df.to_csv('result/result.csv', index = False)

        return results 