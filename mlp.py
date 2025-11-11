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

    input_units_number = patterns.shape
    return targets,input_units_number

#can choose between different extraction methods for the weights/biases
def create_random_extractor(method):
    if(method == "standard"):
        def extractor_function():
            return random.uniform(-0.7,0.7)
        return extractor_function



#=============================
# Neural Network
#=============================

class Neuron:
    """ Represents a single neuron in the neural network. """
    def __init__(self):
        self.bias = extractor()
        self.weights = []
    
class NeuronLayer:
    """ Represents a layer of neurons in the neural network. """
    def __init__(self, neurons):
        self.neurons = [Neuron() for _ in range(neurons)]
        self.num_neurons = neurons

class NeuralNetwork:
    """ Represents a multi-layer perceptron neural network. """
    def __init__(self, num_inputs, num_outputs, neurons_per_layer):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs


        self.neurons_per_layer = neurons_per_layer #neurons per HIDDEN layer
        self.hidden_layers_number = len(neurons_per_layer) #how many hidden layers
        self.hidden_layers = []
        self.output_layer = []#TODO più in la col training valutare se ha senso accorpare hidden layers con input e output in un unica struttura layers



        #initialize hidden weights in a loop
        for i in range(self.hidden_layers_number):
            num_neurons = self.neurons_per_layer[i]
            #only hidden weights number can be chosen
            num_inputs = self.num_inputs if i == 0 else self.neurons_per_layer[i-1] 

            layer = NeuronLayer(num_neurons)

            self.init_weights(layer,num_inputs)
            self.hidden_layers.append(layer)

        output_layer = NeuronLayer(num_outputs)
        self.init_weights(output_layer, self.hidden_layers[-1].num_neurons)
        self.output_layer = output_layer

    def init_weights(self,layer,num_prev_inputs):
        # weight_num = 0
        for neuron in layer.neurons:
            for i in range(num_prev_inputs):
                neuron.weights.append(extractor())
        # weight_num = 0
        # for neuron in 
    def print_network(self):
        print(f'the net has {self.num_inputs} input neurons')
        print(f'the net has {len(self.neurons_per_layer)} hidden layers')
        for i,layer in enumerate(self.hidden_layers):
            print(f'hidden layer {i} weights:')
            for neuron in layer.neurons:
                print(neuron.weights) 
        print(f'the net has {self.num_outputs} output neurons')
        print(f'weights connecting last hidden layer to output:')
        for neuron in self.output_layer.neurons:
            print(neuron.weights)




            




#=============================
# Activation Functions
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



#=============================
# Initialization Flags
#=============================




if __name__ == "__main__":

    config = load_config_json("config.json")
    one_hot_encoding,targets,input_units_number = preprocess_monk()
    # print(input_units_number,config["architecture"]["output_units"],config["architecture"]["neurons_per_layer"])
    extractor = create_random_extractor(config["initialization"]["method"])

    mlp = NeuralNetwork(input_units_number,config["architecture"]["output_units"],config["architecture"]["neurons_per_layer"])


    mlp.print_network()
    # one_hot_encoding,targets,input_units_number = preprocess_exam_file()




