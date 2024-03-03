from utils.generator import WindowGenerator
from model.model import LSTM_Model

import os
import numpy as np
import pandas as pd

INPUT_SIZE = 50
NUM_FEATURES = 4
LABEL_SIZE = 1
UNITS = 64

STOCK_EXCHANGES = os.listdir('./data/raw_data')

def getStockExchangeList():
    return os.listdir('./data/raw_data')

def getDataset():
    df = {i:{} for i in STOCK_EXCHANGES}
    
    for exchange in STOCK_EXCHANGES:
        exchange_directory = f"./data/processed_data/{exchange}"
        for file in os.listdir(exchange_directory):
            stock = file[:-4]
            df[exchange][stock] = pd.read_csv(f"{exchange_directory}/{file}")
    return df

def getTrainingData(input_size = INPUT_SIZE, label_size = LABEL_SIZE, num_features = NUM_FEATURES):
    
    generator = WindowGenerator(input_size, label_size)
    
    df = getDataset()

    training_data = {i:{} for i in STOCK_EXCHANGES}

    for exchange in STOCK_EXCHANGES:
        for stock in df[exchange].keys():
            training_data[exchange][stock] = df[exchange][stock][['Open', 'High', 'Low', 'Close']].values.tolist()
    
    input_data = {i: [] for i in STOCK_EXCHANGES}
    label_data = {i: [] for i in STOCK_EXCHANGES}

    for i in STOCK_EXCHANGES: 
        stock_list = df[i].keys()
        for j in stock_list:
            input_vector, label_vector = generator.generateWindow(training_data[i][j], num_features)
            input_data[i].append(input_vector)
            label_data[i].append(label_vector)

    for i in STOCK_EXCHANGES:
        input_data[i] = np.concatenate(input_data[i])
        label_data[i] = np.concatenate(label_data[i])
    return input_data, label_data

def getPretrainedModel():
    directory = './model/results/'

    model = {}

    for i in STOCK_EXCHANGES:
        model[i] = LSTM_Model(INPUT_SIZE, NUM_FEATURES, UNITS)

    for i in STOCK_EXCHANGES:
        model[i].build(input_shape=(None, INPUT_SIZE, NUM_FEATURES))
        model[i].load_weights(f"{directory}{i}_model.h5")
        model[i].eval = True

    return model