import numpy as np

class WindowGenerator():
    def __init__(self, input_size, label_size, shift=1):
        self.input_size = input_size
        self.label_size = label_size
        self.shift = shift
    
    def generateIndices(self, seq_length):
        input_indices = []
        label_indices = []
        for i in range(seq_length):
            if i + self.input_size + self.shift - 1 >= seq_length:
                break
            else:
                input_indices.append(list(range(i, i + self.input_size)))
                label_indices.append(i + self.input_size + self.shift - 1)
        return input_indices, label_indices
    
    def generateWindow(self, data, n_features):
        numpy_data = np.array(data)
        input_indices, label_indices = self.generateIndices(numpy_data.shape[0])
        input_data = numpy_data[input_indices]
        label_data = numpy_data[label_indices]
        return input_data.reshape((input_data.shape[0], self.input_size, n_features)), label_data