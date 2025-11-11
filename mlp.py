import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def preprocess_monk(type):
    """ Preprocess the MONK dataset by performing one-hot encoding on categorical features.  Returns the one-hot encoded patterns, targets and the number of input units."""
    if type == "train":
        data = pd.read_csv(config["paths"]["train_data"], sep=r"\s+", header=None)
        # print(data.head())
    elif type == "test":
        data = pd.read_csv(config["paths"]["test_data"], sep=r"\s+", header=None)
        # print(data.head())
    else:
        raise ValueError("Invalid type. Must be 'train' or 'test'.")

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

    input_units_number = patterns.shape[0]
    return targets,input_units_number

#can choose between different extraction methods for the weights/biases
def create_random_extractor(method):
    if(method == "standard"):
        def extractor_function():
            return random.uniform(-0.7,0.7)
        return extractor_function
    else:
        raise ValueError("Invalid method.")


#=============================
# Neural Network
#=============================

class Neuron:
    """ Represents a single neuron in the neural network."""
    def __init__(self, bias):
        self.bias = bias
        self.weights = []
    
class NeuronLayer:
    """ Represents a layer of neurons in the neural network."""
    def __init__(self, neurons):
        self.bias = extractor()
        self.neurons = [Neuron(self.bias) for _ in range(neurons)]
        self.num_neurons = neurons

class NeuralNetwork:
    """ Represents a multi-layer perceptron neural network."""
    def __init__(self, num_inputs, num_outputs, neurons_per_layer, training_hyperpar):
        self.num_inputs = num_inputs
        self.neurons_per_layer = neurons_per_layer                      #neurons per HIDDEN layer
        self.hidden_layers_number = len(neurons_per_layer)              #how many hidden layers
        self.num_outputs = num_outputs

        #TODO più in la col training valutare se ha senso accorpare hidden layers con input e output in un unica struttura layers

        self.hidden_layers = []
        self.output_layer = []

        # initialize hidden neuron and weights in a loop
        for i in range(self.hidden_layers_number):
            num_neurons = self.neurons_per_layer[i]
            #only hidden weights number can be chosen
            num_inputs = self.num_inputs if i == 0 else self.neurons_per_layer[i-1]

            hidden_layer = NeuronLayer(num_neurons)
            self.init_weights(hidden_layer,num_inputs)
            self.hidden_layers.append(hidden_layer)

        # initialize output neuron and weights
        output_layer = NeuronLayer(num_outputs)
        self.init_weights(output_layer, self.hidden_layers[-1].num_neurons)
        self.output_layer = output_layer
        
        self.learning_rate = training_hyperpar[0]
        self.momentum = training_hyperpar[1]
        self.batch_size = training_hyperpar[2]

    def init_weights(self,layer,num_prev_inputs):
        for neuron in layer.neurons:
            for i in range(num_prev_inputs):
                neuron.weights.append(extractor())

    def __repr__(self):
        result = []
        result.append(f'The net has {self.num_inputs} input neurons')
        result.append(f'The net has {self.neurons_per_layer} hidden neurons')
        for i, layer in enumerate(self.hidden_layers):
            result.append(f'\thidden layer {i} -> weights:')
            for neuron in layer.neurons:
                result.append(f'\t\t{neuron}:'+'\t'.join(f'{weight:.2f}' for weight in neuron.weights))
        result.append(f'The net has {self.num_outputs} output neurons')
        result.append(f'\toutput layer -> weights:')
        for neuron in self.output_layer.neurons:
            result.append(f'\t\t{neuron}:'+'\t'.join(f'{weight:.2f}' for weight in neuron.weights))
        return '\n'.join(result)


#=============================
# Activation & Loss Functions
#=============================

def sigmoid(x, a):
    return 1 / (1 + np.exp(-x*a))

def tanh(x, a):
    return np.tanh(x*a)

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    return np.log(1 + np.e**x)

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
# TODO


#=============================
# PLOT (da eliminare prima di mandare a Micheli)
#=============================

def gather_weights(nn):
        layers = nn.hidden_layers + [nn.output_layer]
        all_weights = []
        for layer in layers:
            for neuron in layer.neurons:
                all_weights.extend(neuron.weights)
        return all_weights

def plot_network(nn, figsize=(10,6), weight_scaling=5.0, show_bias=True):
    """
    Disegna una rappresentazione grafica dello stato della rete:
    - nodi per input / neuroni dei layer nascosti / output
    - archi con spessore e colore in base al valore dei pesi
    - bias visualizzati vicino ai neuroni (opzionale)
    """
    # layer sizes (including input layer)
    layer_sizes = [nn.num_inputs] + [len(l.neurons) for l in nn.hidden_layers] + [len(nn.output_layer.neurons)]
    n_layers = len(layer_sizes)

    # positions per layer
    positions = []
    for i, size in enumerate(layer_sizes):
        x = i
        if size == 1:
            ys = [0.5]
        else:
            ys = [j/(size-1) for j in range(size)]
        positions.append([(x, y) for y in ys])

    # prepare weight statistics for scaling
    all_w = gather_weights(nn)
    max_abs_w = max((abs(w) for w in all_w), default=1.0)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.5, n_layers-0.5)
    ax.set_ylim(-0.1, 1.1)
    ax.axis('off')

    # draw edges
    layers_obj = nn.hidden_layers + [nn.output_layer]  # corresponds to connections from previous layer -> this layer
    for li, layer in enumerate(layers_obj, start=1):
        prev_pos = positions[li-1]
        cur_pos = positions[li]
        for ni, neuron in enumerate(layer.neurons):
            for pi, w in enumerate(neuron.weights):
                start = prev_pos[pi]
                end = cur_pos[ni]
                norm_w = w / max_abs_w if max_abs_w != 0 else 0
                color = (0, 0.2, 0.8) if w >= 0 else (0.8, 0.1, 0.1)
                lw = max(0.3, abs(norm_w) * weight_scaling)
                ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=lw, alpha=0.7)

    # draw nodes
    for li, layer_pos in enumerate(positions):
        xs = [p[0] for p in layer_pos]
        ys = [p[1] for p in layer_pos]
        if li == 0:
            ax.scatter(xs, ys, s=120, facecolors='lightgray', edgecolors='k', zorder=5)
            for i, (x,y) in enumerate(layer_pos):
                ax.text(x-0.05, y, f"i{i}", ha='right', va='center', fontsize=8)
        else:
            ax.scatter(xs, ys, s=220, facecolors='white', edgecolors='k', zorder=6)
            layer_obj = layers_obj[li-1]
            for ni, (x,y) in enumerate(layer_pos):
                if show_bias:
                    bias = layer_obj.neurons[ni].bias
                    ax.text(x, y+0.08, f"b={bias:.2f}", ha='center', va='top', fontsize=7, color='green')

    ax.set_title("Stato rete neurale: pesi (colore) e bias (testo)")
    plt.tight_layout()
    plt.show()

def plot_weight_histogram(nn, bins=40, figsize=(6,3)):
    all_w = gather_weights(nn)
    plt.figure(figsize=figsize)
    plt.hist(all_w, bins=bins, color='steelblue', edgecolor='k', alpha=0.8)
    plt.title("Distribuzione dei pesi")
    plt.xlabel("Valore peso")
    plt.ylabel("Frequenza")
    plt.grid(alpha=0.3)
    plt.show()

# Utility to display both
def show_network_state(nn):
    plot_network(nn)
    plot_weight_histogram(nn)


#=============================
# Main
#=============================

if __name__ == "__main__":

    config = load_config_json("config.json")

    # MONK DATASET
    one_hot_encoding_train,targets_train,input_units_number = preprocess_monk("train")
    one_hot_encoding_test,targets_test,_ = preprocess_monk("test")

    # # EXAM DATASET
    # one_hot_encoding,targets,input_units_number = preprocess_exam_file()

    hidden_act_func = config["functions"]["hidden"]
    output_act_func = config["functions"]["output"]
    act_func = [hidden_act_func, output_act_func]
    
    training_hyperpar = [config["training"]["learning_rate"], config["training"]["momentum"], config["training"]["batch_size"]]

    extractor = create_random_extractor(config["initialization"]["method"])

    nn = NeuralNetwork(num_inputs=input_units_number,
                       num_outputs=config["architecture"]["output_units"],
                       neurons_per_layer=config["architecture"]["neurons_per_layer"],
                       training_hyperpar=training_hyperpar)

    print(nn)
    
    # show the network state for the just-created network (da eliminare prima di mandare a Micheli)
    show_network_state(nn)


    # one_hot_encoding,targets,input_units_number = preprocess_exam_file()
