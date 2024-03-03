from utils.datetime import is_weekend
from utils.getter import getDataset, getPretrainedModel, getTrainingData
from utils.loss import rmse
from model.model import LSTM_Model

import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

INPUT_SIZE = 50
NUM_FEATURES = 4
LABEL_SIZE = 1
UNITS = 64

class ModelManager():
    def __init__(self):
        self.df = getDataset()
        self.stock_exchanges = self.df.keys()
        self.model = getPretrainedModel()
        self.input_data, self.label_data = getTrainingData()
        self.last_day, self.min_self.last_day, self.max_self.last_day = self.getDatetimeAttribute()


    def getDatetimeAttribute(self):
        last_day = {i: {} for i in self.stock_exchanges}
        min_last_day = "12/31/2024"
        max_last_day = "1/1/2024"

        for i in self.stock_exchanges:
            for j in list(self.df[i].keys()):
                date = self.df[i][j]["Date"].values[-1]
                last_day[i][j] = date
                if datetime.datetime.strptime(date, "%m/%d/%Y") < datetime.datetime.strptime(self.min_self.last_day, "%m/%d/%Y"):
                    min_last_day = date
                if datetime.datetime.strptime(date, "%m/%d/%Y") > datetime.datetime.strptime(max_last_day, "%m/%d/%Y"):
                    max_last_day = date

        return self.last_day, min_last_day, max_last_day
    
    def train(self):
        for i in self.stock_exchanges:
            self.model[i] = LSTM_Model(INPUT_SIZE, NUM_FEATURES, UNITS)
            self.model[i].compile(optimizer=Adam(0.0025), loss=rmse)

        with tf.device('/GPU:0'):
            for i in self.stock_exchanges:
                checkpoint_path = f"{i}_model.h5"
                checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path,
                                            save_weights_only=True,
                                            save_best_only=True,
                                            save_freq='epoch')
                self.model[i].fit(self.input_data[i], self.label_data[i],
                                        epochs = 250,
                                        batch_size = 64,
                                        verbose = 1,
                                        validation_split = 0.1,
                                        shuffle = True,
                                        callbacks=[checkpoint_callback])

    def stock_predict(self, stock_name, stock_exchange, date):
        try:
            stock_df = self.df[stock_exchange][stock_name]
        except KeyError:
            return f"Stock {stock_name} not found."

        last_records = stock_df.tail(50)
        last_date = self.last_day[stock_exchange][stock_name]

        predictor = self.model[stock_exchange]

        input_size = last_records.shape[0]
        num_features = 4

        input_data = np.array(last_records[['Open', 'High', 'Low', 'Close']]).reshape((1, input_size, num_features))
        current_date = datetime.datetime.strptime(last_date, "%m/%d/%Y") + datetime.timedelta(days=1)
        predicted_data = None
        while current_date.strftime("%d/%m/%Y") != date:
            if is_weekend(current_date):
                current_date += datetime.timedelta(days=1)
                continue
            predicted_data = predictor.predict(input_data, verbose = 0)
            input_data = np.concatenate((input_data[:,1:,:], predicted_data.reshape(1, 1, 4)), axis = 1)
            current_date += datetime.timedelta(days=1)
        if predicted_data is None:
            return "Invalid date!"
        return predicted_data
    
    def exchange_predict(self, stock_exchange, date):
        if datetime.datetime.strptime(date, "%d/%m/%Y") <= datetime.datetime.strptime(self.max_last_day, "%m/%d/%Y"):
            return "Invalid date!"

        last_records = {}

        for i in self.df[stock_exchange].keys():
            last_records[i] = np.array(self.df[stock_exchange][i].tail(50))[:,4:8].astype('float32')

        try:
            predictor = self.model[stock_exchange]
        except KeyError:
            return f"Stock exchange {stock_exchange} not found."

        input_size = 50
        num_features = 4

        current_date = datetime.datetime.strptime(self.min_self.last_day, "%m/%d/%Y") + datetime.timedelta(days=1)

        while current_date.strftime("%d/%m/%Y") != date:
            if is_weekend(current_date):
                current_date += datetime.timedelta(days=1)
                continue
            for stock_symbol, records in last_records.items():
                if current_date.strftime("%m/%d/%Y") < self.last_day[stock_exchange][stock_symbol]:
                    continue
                input_data = records[-input_size:].reshape((1, input_size, num_features))
                prediction = predictor.predict(input_data, verbose=0)
                last_records[stock_symbol] = np.concatenate((records, prediction.reshape(1, 4)))
            current_date += datetime.timedelta(days=1)

        stock_names = []
        last_date = []
        last_close_price = []
        current_date = []
        predict_close_price = []
        direction = []
        profit = []

        for stock_symbol, records in last_records.items():
            stock_names.append(stock_symbol)
            last_date.append(self.last_day[stock_exchange][stock_symbol])
            last_close_price.append(last_records[stock_symbol][50][3])
            current_date.append(date)
            predict_close_price.append(last_records[stock_symbol][-1][3])
            direction.append('Increase' if last_records[stock_symbol][-1][3] > last_records[stock_symbol][50][3] else 'Decrease')
            profit.append(last_records[stock_symbol][-1][3] - last_records[stock_symbol][50][3])

        data = {
            'Stock Names': stock_names,
            'Last Date': last_date,
            'Last Close Price': last_close_price,
            'Current Date': current_date,
            'Predicted Close Price': predict_close_price,
            'Direction': direction,
            'Profit': profit
        }
        return pd.DataFrame(data).sort_values(by = ["Profit"], ascending = False).head(10)

    def plot(self, stock_name, stock_exchange, date):

        try:
            stock_df = self.df[stock_exchange][stock_name]
        except KeyError:
            return f"Stock {stock_name} not found."
        
        last_records = stock_df.tail(50)
        last_date = self.last_day[stock_exchange][stock_name]

        predictor = self.model[stock_exchange]

        input_size = last_records.shape[0]
        num_features = 4

        input_data = np.array(last_records[['Open', 'High', 'Low', 'Close']]).reshape(1, input_size, num_features)

        future_data = [input_data[:,-1,:].reshape(1, num_features)]

        current_date = datetime.datetime.strptime(last_date, "%m/%d/%Y") + datetime.timedelta(days=1)
        predicted_data = None
        while current_date.strftime("%d/%m/%Y") != date:
            if is_weekend(current_date):
                current_date += datetime.timedelta(days=1)
                continue
            predicted_data = predictor.predict(input_data, verbose = 0)
            future_data.append(predicted_data.reshape(1, num_features))
            input_data = np.concatenate((input_data[:,1:,:], predicted_data.reshape(1, 1, num_features)), axis = 1)
            current_date += datetime.timedelta(days=1)
        if predicted_data is None:
            return "Invalid date!"

        future_data = np.concatenate(future_data)

        plt.plot(range(stock_df.shape[0]), stock_df['Open'], label='Open Price Original')
        plt.plot(range(stock_df.shape[0]), stock_df['Close'], label='Close Price Original')
        plt.plot(range(stock_df.shape[0] - 1, stock_df.shape[0] + future_data.shape[0] - 1), future_data[:,0], label='Open Price Predict')
        plt.plot(range(stock_df.shape[0] - 1, stock_df.shape[0] + future_data.shape[0] - 1), future_data[:,3], label='Close Price Predict')

        plt.title(f'Open and close price of {stock_name} on {stock_exchange}')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.xticks([])

        plt.legend()
        plt.show()
    