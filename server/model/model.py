from keras import Model
from keras.layers import Dense, Flatten
from keras.layers import BatchNormalization
from keras.layers import LSTM

class LSTM_Model(Model):
    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun
    
    def __init__(self, input_size, num_features, units):
        super().__init__()
        
        # input_shape = (input_size, num_features)
        self.input_size = input_size
        self.num_features = num_features
        self.units = units
        
        self.flatten = Flatten()
        
        self.normalization = BatchNormalization(axis = 1)
        
        self.lstm1 = LSTM(
            self.units,
            input_shape = (self.input_size, self.num_features),
            return_sequences = True,
            recurrent_initializer='glorot_uniform'
        )
        
        self.lstm2 = LSTM(
            self.units,
            input_shape = (self.input_size, self.num_features),
            return_sequences = False,
            recurrent_initializer='glorot_uniform'
        )
        
        self.dense_1 = Dense(int(self.units/2), activation = 'swish', name = 'dense_1')
        self.dense_2 = Dense(int(self.units/2), activation = 'swish', name = 'dense_2')
        self.dense_3 = Dense(self.num_features, name = 'dense_3')
    
    def call(self, x):
        x = self.normalization(x)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        return x