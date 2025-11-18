from utils import plot_network # (da eliminare prima di mandare a Micheli)
import activations as actfun
import numpy as np

np.random.seed(42)

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

        self.inputs = None
        self.net = None
        self.outputs = None

        self.deltas = None

        self.weights_grad_acc = None
        self.bias_grad_acc = None

class NeuralNetwork:
    """ Represents a multi-layer perceptron neural network. """
    def __init__(self, num_inputs, num_outputs, neurons_per_layer, training_hyperpar, extractor, activation=["relu", "sigmoid"]):
        self.extractor = extractor
        self.hidden_activation = actfun.activation_functions[activation[0]][0]
        self.d_hidden_activation = actfun.activation_functions[activation[0]][1]
        self.output_activation = actfun.activation_functions[activation[1]][0]
        self.d_output_activation = actfun.activation_functions[activation[1]][1]

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

        self.layers = self.hidden_layers + [self.output_layer]
        

        self.learning_rate = training_hyperpar["learning_rate"]
        self.momentum = training_hyperpar["momentum"]
        # self.batch_size = training_hyperpar[2] TODO rimuovere se non serve (o nel caso leggere il valore corretto)

        self.vl=training_hyperpar["vl"]

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
            layer.inputs = current_inputs
            layer.net = layer.weights @ layer.inputs + layer.biases
            layer.outputs = self.hidden_activation(layer.net)
            current_inputs = layer.outputs

        # output layer
        self.output_layer.inputs = current_inputs
        self.output_layer.net = self.output_layer.weights @ self.output_layer.inputs + self.output_layer.biases
        self.output_layer.outputs = self.output_activation(self.output_layer.net)

        return self.output_layer.outputs

    def back_prop(self, target):
        previous_delta = None
        previous_weights = None
        for layer in reversed(self.layers):
            # output layer
            if layer == self.output_layer:
                layer.deltas = (target - layer.outputs) * self.d_output_activation(layer.net)   #TODO guardare se le formule di attivazione sono corrette                 # delta_k
            # hidden layers
            elif layer in self.hidden_layers:
                layer.deltas = (previous_weights.T @ previous_delta) * self.d_hidden_activation(layer.net)          # delta_j
            else:
                raise Exception("Layer not recognized")
            
            previous_delta, previous_weights = layer.deltas, layer.weights


    def weights_update(self,batch,l):
        if(batch == "full" or batch == "minibatch"):
            for layer in self.layers:
                layer.weights = (layer.weights + self.learning_rate * (layer.weights_grad_acc/l)) 
                layer.biases = (layer.biases + self.learning_rate * (layer.bias_grad_acc/l)) 
        else:

            for layer in self.layers:
                layer.weights = layer.weights + self.learning_rate * np.outer(layer.deltas, layer.inputs)
                layer.biases = layer.biases + self.learning_rate * layer.deltas 

    #TODO usare early stopping o comunque altri parametri per capire dopo quante epoche fermarsi

    def train(self, X, T,train_args):
        batch = train_args["batch_type"]
        batch_size = train_args["batch_size"]
        epochs = train_args["epochs"]
        for i in range(epochs):
            if(batch == "full"):
                #for batch gradient descent, i need to have a way to accumulate the gradient for each pattern with a shape like the weights
                for layer in self.layers: 
                    layer.weights_grad_acc = np.zeros_like(layer.weights)
                    layer.bias_grad_acc = np.zeros_like(layer.biases)

                #iterate on eac pattern ff and backprop
                for x,t in zip(X,T):
                    o = self.feed_forward(x)
                    deltas = self.back_prop(t)
                    
                    #accumulate gradient
                    for layer in self.layers:
                        layer.weights_grad_acc += np.outer(layer.deltas, layer.inputs)
                        layer.bias_grad_acc += layer.deltas

                self.weights_update(batch,len(X))

            if(batch == 'minibatch'):
                for layer in self.layers: 
                    layer.weights_grad_acc = np.zeros_like(layer.weights)
                    layer.bias_grad_acc = np.zeros_like(layer.biases)

                counter = 0
                for x,t in zip(X,T):
                    self.feed_forward(x)
                    self.back_prop(t)

                    #using a counter manages the istances if dataset ends before batch size is reached
                    counter += 1

                    for layer in self.layers:
                        layer.weights_grad_acc += np.outer(layer.deltas, layer.inputs)
                        layer.bias_grad_acc += layer.deltas
                
                    if(counter == batch_size):    
                        self.weights_update(batch,counter)
                        for layer in self.layers: 
                            layer.weights_grad_acc = np.zeros_like(layer.weights)
                            layer.bias_grad_acc = np.zeros_like(layer.biases)
                        counter = 0

                #flush update if necessary
                if(counter!=0):
                    self.weights_update(batch,counter)                
            
            else:
                np.random.shuffle(X)
                np.random.shuffle(T)
                for x, t in zip(X,T):
                    # for _ in range(5):
                    o = self.feed_forward(x)
                    delta = self.back_prop(t)
                    self.weights_update(0,0)
                    # print(o)
                    o = self.feed_forward(x)
                    # print(o)
                    # print(t)
                    # print('-'*30)

    # def validate()
