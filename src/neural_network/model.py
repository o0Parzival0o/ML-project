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
        self.weights = None
        self.biases = None
        self.outputs = None

class NeuralNetwork:
    """ Represents a multi-layer perceptron neural network. """
    def __init__(self, num_inputs, num_outputs, neurons_per_layer, training_hyperpar, extractor, activation=["relu", "sigmoid"]):
        self.extractor = extractor
        self.hidden_activation = actfun.activation_functions[activation[0]]
        self.output_activation = actfun.activation_functions[activation[1]]

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

            hidden_layer.weights = np.array([neuron.weights for neuron in hidden_layer.neurons])
            hidden_layer.biases = np.array([neuron.bias for neuron in hidden_layer.neurons])

        # initialize output neuron and weights
        self.output_layer = NeuronLayer(num_outputs)
        self.init_weights(self.output_layer, self.hidden_layers[-1].num_neurons)

        self.output_layer.weights = np.array([neuron.weights for neuron in self.output_layer.neurons])
        self.output_layer.biases = np.array([neuron.bias for neuron in self.output_layer.neurons])
        

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
            current_inputs = self.hidden_activation(layer.weights @ current_inputs + layer.biases)

        # output layer    
        outputs = self.output_activation(self.output_layer.weights @ current_inputs + self.output_layer.biases)

        return outputs

    # def back_prop(self, target, d_act_func):
    #     delta_weights = []
    #     for layer in self.layers:
    #         previous_delta = None
    #         net = np.sum(layer.weights @ layer.outputs)
    #         if layer == self.output_layer:
    #             delta_k = (target - layer.outputs) * d_act_func(net)
    #             delta_weights.append(delta_k*layer.outputs)
    #         elif layer in self.hidden_layers:
    #             delta_j = np.sum(previous_delta * layer.weights) * d_act_func(net)
    #         else:
    #             raise Exception("Layer not recognized")
    #         previous_delta = delta_weights