import pandas as pd
import json
import random

def load_config_json(filepath):
    """Loads the configuration from the specified JSON file."""
    with open(filepath, 'r') as f:
        config = json.load(f)
    return config


class Neuron:
    def __init__(self,bias):
        self.bias = bias
        self.weights = []
    


class NeuronLayer:
    def __init__(self,neurons):
        self.bias = random.random()
        self.neurons = [Neuron(self.bias) for _ in range(neurons)]




class NeuralNetwork:
    def __init__(self,num_inputs,num_hidden,num_outputs,neurons_per_layer):
        self.num_inputs = num_inputs
        # self.hidden = NeuronLayer()#TODO valutare se ha senso implementare un diverso numero di neuroni per hidden layer
        # self.num_outputs= num_outputs

        self.hidden_layers = []

        self.output_layer = NeuronLayer(num_outputs)


        for neurons in neurons_per_layer:
            self.hidden_layers.append(NeuronLayer(neurons))


        

def preprocess_monk():
    data = pd.read_csv(config["paths"]["train_data"], sep=r"\s+", header=None)

    # print(data.head())

    # the first column in the dataset is the target, the remaining values are the pattern
    #TODO vedere se c'è da rimuovere i nomi pure qua 
    targets = data[data.columns[0]]

    # print(targets.head(25))

    #removing target and name
    patterns = data.drop(data.columns[0], axis=1)
    patterns = patterns.drop(data.columns[7], axis=1)#TODO capire come adattare esigenze specifiche del dataset ad altri casi generali (o se fare due versioni separate per dataset)
    # print(patterns.head(5))

    #this is necessary for pandas to get the dummies otherwise it isn't happy
    patterns = patterns.astype(str)


    one_hot_encoding = pd.get_dummies(patterns, prefix=['A1','A2','A3','A4','A5','A6'])

    # print(one_hot_encoding)

    input_units_number = one_hot_encoding.shape[1]
    return one_hot_encoding,targets,input_units_number



config = load_config_json("config.json")
one_hot_encoding,targets,input_units_number = preprocess_monk()



