from model.model import LSTM_Model

INPUT_SIZE = 50
NUM_FEATURES = 4
LABEL_SIZE = 1
UNITS = 64

STOCK_EXCHANGES = ["HNX", "HOSE", "UPCOM"]

def getPretrainedModel():
    directory = 'model/results/'

    model = {}

    for i in STOCK_EXCHANGES:
        model[i] = LSTM_Model(INPUT_SIZE, NUM_FEATURES, UNITS)

    for i in STOCK_EXCHANGES:
        model[i].build(input_shape=(None, INPUT_SIZE, NUM_FEATURES))
        model[i].load_weights(f"{directory}{i}_model.h5")
        model[i].eval = True

    return model