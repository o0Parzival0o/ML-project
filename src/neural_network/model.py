from utils import plot_network # (da eliminare prima di mandare a Micheli)
import activations as actfun
import numpy as np


class Neuron:
    """ Represents a single neuron in the neural network. """
    def __init__(self):
        self.bias = None
        self.weights = []
    
class NeuronLayer:
    """ Represents a layer of neurons in the neural network. """
    def __init__(self, neurons):
        self.neurons = [Neuron() for _ in range(neurons)]
        self.num_neurons = neurons

class NeuralNetwork:
    """ Represents a multi-layer perceptron neural network. """
    def __init__(self, num_inputs, num_outputs, neurons_per_layer, training_hyperpar, extractor, activation="sigmoid"):
        self.extractor = extractor
        self.activation = actfun.activation_functions[activation]

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
            self.init_weights(hidden_layer, num_inputs)
            self.hidden_layers.append(hidden_layer)

        # initialize output neuron and weights
        output_layer = NeuronLayer(num_outputs)
        self.init_weights(output_layer, self.hidden_layers[-1].num_neurons)
        self.output_layer = output_layer
        
        self.learning_rate = training_hyperpar[0]
        self.momentum = training_hyperpar[1]
        self.batch_size = training_hyperpar[2]

    def init_weights(self, layer, num_prev_inputs):
        for neuron in layer.neurons:
            neuron.bias = self.extractor()
            for _ in range(num_prev_inputs):
                neuron.weights.append(self.extractor())

    def __repr__(self):
        result = []
        result.append(f'The net has {self.num_inputs} input neurons')
        result.append(f'The net has {self.neurons_per_layer} hidden neurons')
        for i, layer in enumerate(self.hidden_layers):
            result.append(f'\thidden layer {i} -> weights:')
            for neuron in layer.neurons:
                result.append(f'\t\t{neuron}:\t'+'\t'.join(f'{weight:.2f}' for weight in neuron.weights))
        result.append(f'The net has {self.num_outputs} output neurons')
        result.append(f'\toutput layer -> weights:')
        for neuron in self.output_layer.neurons:
            result.append(f'\t\t{neuron}:\t'+'\t'.join(f'{weight:.2f}' for weight in neuron.weights))
        return '\n'.join(result)
    
    def plot(self): # (da eliminare prima di mandare a Micheli)
        plot_network(self)

    def feed_forward(self, inputs):
        current_inputs = inputs

        # hidden layers
        for layer in self.hidden_layers:
            weights = np.array([neuron.weights for neuron in layer.neurons])
            biases = np.array([neuron.bias for neuron in layer.neurons])
        
            current_inputs = self.activation(np.dot(weights, current_inputs) + biases)

        # output layer
        weights = np.array([neuron.weights for neuron in self.output_layer.neurons])
        biases = np.array([neuron.bias for neuron in self.output_layer.neurons])
    
        outputs = self.activation(np.dot(weights, current_inputs) + biases)

        return np.array(outputs)
