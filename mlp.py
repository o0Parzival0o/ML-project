import pandas as pd
import numpy as np

import json
import random

#=============================
# Utility
#=============================

def load_config_json(filepath):
    """Loads the configuration from the specified JSON file."""
    with open(filepath, 'r') as f:
        config = json.load(f)
    return config

def preprocess_monk():
    """ Preprocess the MONK dataset by performing one-hot encoding on categorical features.  Returns the one-hot encoded patterns, targets and the number of input units."""
    data = pd.read_csv(config["paths"]["train_data"], sep=r"\s+", header=None)
    # print(data.head())

    # the first column in the dataset is the target, the remaining values are the pattern
    targets = data[data.columns[0]]
    # print(targets.head(25))

    # removing target and name
    patterns = data.drop(data.columns[0], axis=1)
    patterns = patterns.drop(data.columns[-1], axis=1)
    # print(patterns.head(5))

    # this is necessary for pandas to get the dummies otherwise it isn't happy
    patterns = patterns.astype(str)

    one_hot_encoding = pd.get_dummies(patterns, prefix=['A1','A2','A3','A4','A5','A6'])
    # print(one_hot_encoding)

    input_units_number = one_hot_encoding.shape[1]
    return one_hot_encoding,targets,input_units_number

def preprocess_exam_file():
    """ Preprocess the exam dataset.  Returns the targets and the number of input units."""
    data = pd.read_csv(config["paths"]["exam_file"], sep=r"\s+", header=None)

    # the lasts 4 columns in the dataset are the targets, the remaining values are the pattern and examples index
    targets = data[data.columns[-4:]]
    # print(targets.head(25))

    #removing target and name
    patterns = data.drop(data.columns[-4:], axis=1)
    patterns = patterns.drop(data.columns[0], axis=1)
    # print(patterns.head(5))

    input_units_number = patterns.shape[1]
    return targets,input_units_number


#=============================
# Neural Network
#=============================

class Neuron:
    """ Represents a single neuron in the neural network. """
    def __init__(self):
        self.bias = random.uniform(-1, 1)
        self.weights = []
    
class NeuronLayer:
    """ Represents a layer of neurons in the neural network. """
    def __init__(self, num_neurons, num_inputs_per_neuron):
        self.neurons = []
        for _ in range(num_neurons):
            neuron = Neuron()
            neuron.weights = [random.uniform(-1, 1) for _ in range(num_inputs_per_neuron)]
            self.neurons.append(neuron)

class NeuralNetwork:
    """ Represents a multi-layer perceptron neural network. """
    def __init__(self, num_inputs,  neurons_per_layer, num_outputs):
        # hidden layers
        input_size = num_inputs
        self.hidden_layers = []

        for neurons in neurons_per_layer:
            layer = NeuronLayer(neurons, input_size)
            self.hidden_layers.append(layer)
            input_size = neurons
        
        # output layer
        self.output_layer = NeuronLayer(num_outputs, input_size)


        def init_weights_from_inputs_to_hidden_layers(self):
            weight_num = 0                                  #TODO continuare da qui
        


#=============================
# Activation & Loss Functions
#=============================

def sigmoid(x, a):
    return 1 / (1 + np.exp(-x*a))

def tanh(x, a):
    return np.tanh(x*a)

def relu(x):
    return np.maximum(0, x)

def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def tanh_like(x, n):
    return np.sign(x) * (1 + ((2**n * np.abs(x) - np.floor(2**n * np.abs(x)))/2 - 1)/(2**(np.floor(2**n * np.abs(x)))))

def MSE(o, t):
    l = t.shape[0]
    return 1/l * np.sum(np.sum((o-t)**2, axis=1), axis=0)

def MEE(o, t):
    l = t.shape[0]
    return 1/l * np.sum(np.sqrt(np.sum((o-t)**2, axis=1)), axis=0)

if __name__ == "__main__":

    config = load_config_json("config.json")
    one_hot_encoding,targets,input_units_number = preprocess_monk()    
    # one_hot_encoding,targets,input_units_number = preprocess_exam_file()