from utils import plot_network # (da eliminare prima di mandare a Micheli)
import activations as actfun
import losses

import numpy as np
import matplotlib.pyplot as plt
import copy

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

        self.bp_deltas = None

        self.delta_weights_old = None
        self.delta_biases_old = None

        self.weights_grad_acc = None
        self.biases_grad_acc = None

class NeuralNetwork:
    """ Represents a multi-layer perceptron neural network. """
    def __init__(self, num_inputs, num_outputs, neurons_per_layer, training_hyperpar, extractor, activation=[["relu", 0.], ["sigmoid", 1.]]):
        self.extractor = extractor
        self.hidden_activation = actfun.activation_functions[activation[0][0]][0]
        self.d_hidden_activation = actfun.activation_functions[activation[0][0]][1]
        self.hidden_activation_param = activation[0][1]
        self.output_activation = actfun.activation_functions[activation[1][0]][0]
        self.d_output_activation = actfun.activation_functions[activation[1][0]][1]
        self.output_activation_param = activation[1][1]

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
            self.init_weights(hidden_layer, num_inputs, self.neurons_per_layer[i])
            self.hidden_layers.append(hidden_layer)

            hidden_layer.weights = np.array([neuron.weights for neuron in hidden_layer.neurons])
            hidden_layer.biases = np.array([neuron.bias for neuron in hidden_layer.neurons])

        # initialize output neuron and weights
        self.output_layer = NeuronLayer(num_outputs)
        self.init_weights(self.output_layer, self.hidden_layers[-1].num_neurons, num_outputs)

        self.output_layer.weights = np.array([neuron.weights for neuron in self.output_layer.neurons])
        self.output_layer.biases = np.array([neuron.bias for neuron in self.output_layer.neurons])

        self.layers = self.hidden_layers + [self.output_layer]

        for layer in self.layers:
            layer.delta_weights_old = np.zeros_like(layer.weights)
            layer.delta_biases_old = np.zeros_like(layer.biases)
        

        self.learning_rate = training_hyperpar["learning_rate"]
        self.min_learning_rate = training_hyperpar["learning_rate"]/100
        self.orig_learning_rate = training_hyperpar["learning_rate"]
        self.momentum = training_hyperpar["momentum"]
        self.regularization = training_hyperpar["regularization"]
        self.decay_factor = training_hyperpar["decay_factor"]

        self.tr_loss = None
        self.tr_accuracy = None
        self.vl_loss = None
        self.vl_accuracy = None
        self.ts_loss = None
        self.ts_accuracy = None

        self.total_iters = 0

    def init_weights(self, layer, num_inputs, num_outputs):
        for neuron in layer.neurons:
            neuron.bias = self.extractor(fan_in=num_inputs, fan_out=num_outputs, a=(self.output_activation_param if layer == self.output_layer else self.hidden_activation_param))
            for _ in range(num_inputs):
                neuron.weights.append(self.extractor(fan_in=num_inputs, fan_out=num_outputs, a=(self.output_activation_param if layer == self.output_layer else self.hidden_activation_param)))

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
            layer.outputs = self.hidden_activation(layer.net, self.hidden_activation_param)
            current_inputs = layer.outputs

        # output layer
        self.output_layer.inputs = current_inputs
        self.output_layer.net = self.output_layer.weights @ self.output_layer.inputs + self.output_layer.biases
        self.output_layer.outputs = self.output_activation(self.output_layer.net, self.output_activation_param)

        return self.output_layer.outputs

    def back_prop(self, target):
        previous_delta = None
        previous_weights = None
        for layer in reversed(self.layers):
            # output layer
            if layer == self.output_layer:
                layer.bp_deltas = (target - layer.outputs) * self.d_output_activation(layer.net, self.output_activation_param)                       # delta_k
            # hidden layers
            elif layer in self.hidden_layers:
                layer.bp_deltas = (previous_weights.T @ previous_delta) * self.d_hidden_activation(layer.net, self.hidden_activation_param)          # delta_j
            else:
                raise Exception("Layer not recognized")
            
            previous_delta, previous_weights = layer.bp_deltas, layer.weights


    def weights_update(self, l):

        self.total_iters += 1

        for layer in self.layers:
            if l == 1:
                delta_weights = self.learning_rate * np.outer(layer.bp_deltas, layer.inputs) + self.momentum * layer.delta_weights_old
                delta_biases = self.learning_rate * layer.bp_deltas + self.momentum * layer.delta_biases_old
                                
            else:
                delta_weights = self.learning_rate * (layer.weights_grad_acc / l) + self.momentum * layer.delta_weights_old
                delta_biases = self.learning_rate * (layer.biases_grad_acc / l) + self.momentum * layer.delta_biases_old
            
            layer.weights = layer.weights + delta_weights - self.regularization * layer.weights
            layer.biases = layer.biases + delta_biases - self.regularization * layer.biases
            
            layer.delta_weights_old = delta_weights
            layer.delta_biases_old = delta_biases
            self.learning_rate = self.orig_learning_rate / (1+self.decay_factor*self.total_iters) if self.learning_rate > self.min_learning_rate else self.learning_rate


    def train(self, X_tr, T_tr, X_vl=None, T_vl=None, train_args=None, loss_func=None, early_stopping = None):

        batch_size = train_args["batch"]["batch_size"]
        batch_droplast = train_args["batch"]["drop_last"]
        max_epochs = train_args["epochs"]

        #Early stopping variables
        early_stopping_cond = early_stopping["enabled"]
        patience = early_stopping["patience"] if early_stopping_cond else None
        monitor = early_stopping["monitor"] if early_stopping_cond else None

        if X_vl is not None:
            best_model_weights = copy.deepcopy(self.layers)

        loss_func = losses.losses_functions[loss_func]
        tr_loss = []
        tr_accuracy = []
        vl_loss = []
        vl_accuracy = []

        tr_loss.append(self.loss_calculator(X_tr, T_tr, loss_func))                 # loss 0: with random parameters
        tr_accuracy.append(self.accuracy_calculator(X_tr, T_tr))

        if X_vl is not None:
            vl_loss.append(self.loss_calculator(X_vl, T_vl, loss_func)) 
            vl_accuracy.append(self.accuracy_calculator(X_vl, T_vl))
        
        self.total_iters = 0
        current_epoch = 0
        patience_index = patience
        while current_epoch < max_epochs and (patience_index > 0 if early_stopping_cond else True):
            current_epoch += 1
            
            if batch_size == "full":
                #for batch gradient descent, i need to have a way to accumulate the gradient for each pattern with a shape like the weights
                for layer in self.layers:
                    layer.weights_grad_acc = np.zeros_like(layer.weights)
                    layer.biases_grad_acc = np.zeros_like(layer.biases)
            
                #iterate on each pattern of and backprop
                for x,t in zip(X_tr, T_tr):
                    self.feed_forward(x)
                    self.back_prop(t)

                    #accumulate gradient
                    for layer in self.layers:
                        layer.weights_grad_acc += np.outer(layer.bp_deltas, layer.inputs)
                        layer.biases_grad_acc += layer.bp_deltas

                self.weights_update(len(X_tr))

            elif isinstance(batch_size, int) and batch_size > 0:
                perm = np.random.permutation(len(X_tr))
                X_tr = X_tr[perm]
                T_tr = T_tr[perm]
                
                if batch_size != 1:
                    for layer in self.layers: 
                        layer.weights_grad_acc = np.zeros_like(layer.weights)
                        layer.biases_grad_acc = np.zeros_like(layer.biases)

                    counter = 0
                    for x,t in zip(X_tr,T_tr):
                        self.feed_forward(x)
                        self.back_prop(t)

                        #using a counter manages the istances if dataset ends before batch size is reached
                        counter += 1

                        for layer in self.layers:
                            layer.weights_grad_acc += np.outer(layer.bp_deltas, layer.inputs)
                            layer.biases_grad_acc += layer.bp_deltas

                        if counter == batch_size:
                            self.weights_update(counter)
                            for layer in self.layers: 
                                layer.weights_grad_acc = np.zeros_like(layer.weights)
                                layer.biases_grad_acc = np.zeros_like(layer.biases)
                            counter = 0

                    #flush update if necessary
                    if counter != 0 and not batch_droplast:
                        self.weights_update(counter)

                elif batch_size == 1: 
                    for x,t in zip(X_tr,T_tr):
                        self.feed_forward(x)
                        self.back_prop(t)
                        self.weights_update(1)


            else:
                raise TypeError('batch_size is not positive int or "full".')

            if X_vl is not None:
                current_vl_loss = self.loss_calculator(X_vl, T_vl, loss_func)
                vl_loss.append(current_vl_loss)
                current_vl_accuracy = self.accuracy_calculator(X_vl, T_vl)
                vl_accuracy.append(current_vl_accuracy)
            
            #check if vl increases
            if not early_stopping_cond:
                pass

            elif early_stopping_cond and monitor == "val_loss":
                
                # "sensibility" of early stopping
                rel_epsilon1 = 0.001
                rel_epsilon2 = 0.002

                if current_vl_loss <= np.min(vl_loss[:-1]) * (1 - rel_epsilon1):
                    patience_index = patience
                    best_model_weights = copy.deepcopy(self.layers)                     ###########

                elif current_vl_loss > np.min(vl_loss[:-1]) * (1 + rel_epsilon2):
                    patience_index -= 1
                
                else:
                    pass

            elif early_stopping_cond and monitor == "val_accuracy":

                # "sensibility" of early stopping
                rel_epsilon1 = 0.0005
                rel_epsilon2 = 0.002

                if current_vl_accuracy >= np.max(vl_accuracy[:-1]) * (1 - rel_epsilon1):
                    patience_index = patience
                    best_model_weights = copy.deepcopy(self.layers)

                elif current_vl_accuracy < np.max(vl_accuracy[:-1]) * (1 + rel_epsilon2):
                    patience_index -= 1
                
                else:
                    pass

            else:
                raise ValueError('monitor parameter must be "val_loss" or "val_accuracy"')

            tr_loss.append(self.loss_calculator(X_tr, T_tr, loss_func))
            tr_accuracy.append(self.accuracy_calculator(X_tr, T_tr))


        print(f"Early stopping at epoch: {current_epoch}" if patience_index == 0 else "Max epoch reached")

        if X_vl is not None:
            self.layers = copy.deepcopy(best_model_weights)

        self.hidden_layers = self.layers[:-1]
        self.output_layer = self.layers[-1]

        self.tr_loss = tr_loss
        self.tr_accuracy = tr_accuracy
        if X_vl is not None:
            self.vl_loss = vl_loss
            self.vl_accuracy = vl_accuracy
    
    def loss_calculator(self, X, T, loss_func):
        predictions = np.array([self.feed_forward(x) for x in X])
        loss = loss_func(predictions, T)
        return loss
    
    def accuracy_calculator(self, X, T):
        correct_predict = 0
        for x,t in zip(X,T):
            predictions = self.feed_forward(x)
            if (self.output_activation.__name__ == "sigmoid" and (predictions >= 0.5 and t == 1 or predictions < 0.5 and t == 0)) or (self.output_activation.__name__ == "tanh" and (predictions >= 0. and t == 1 or predictions < 0. and t == 0)):
                correct_predict += 1
        accuracy = correct_predict / len(T)
        return accuracy

    def test(self, X, T):
        correct_predict = 0
        for x,t in zip(X,T):
            o = self.feed_forward(x)
            if o >= 0.5 and t == 1 or o < 0.5 and t == 0:
                correct_predict += 1
        accuracy = correct_predict/len(T)
        print(f"The model obtained an accuracy of {accuracy:.2%} on test set")
    
    def plot_metrics(self, fig_loss=None, fig_acc=None, rows=1, cols=1, plot_index=0, changing_hyperpar=None, title=None):
        
        if self.tr_loss != None and fig_loss:
            ax_loss = fig_loss.add_subplot(rows, cols, plot_index + 1)
            ax_loss.plot(self.tr_loss, c='r', linestyle='-', label='Training')

            if self.vl_loss != None:
                ax_loss.plot(self.vl_loss, c='b', linestyle='--', label='Validation')

            if changing_hyperpar:
                for k, v in changing_hyperpar.items():
                    param_name = k.split(".")[-1]
                    ax_loss.plot([], [], " ", label=f"{param_name}: {v}")

            if plot_index >= rows * (cols - 1):
                ax_loss.set_xlabel("Epochs")
            if plot_index % cols == 0:
                ax_loss.set_ylabel("Loss / Validation loss" if self.vl_loss else "Loss")

            if self.vl_loss is not None:
                if title == "best_model":
                    ax_loss.set_title(f"Best model (VL: {self.vl_loss[-1]:.4f})", fontsize=8, fontweight='bold')
                else:
                    ax_loss.set_title(f"Trial {plot_index+1} (VL: {self.vl_loss[-1]:.4f})", fontsize=8, fontweight='bold')
            else:
                if title == "best_model":
                    ax_loss.set_title(f"Retraining", fontsize=8, fontweight='bold')
                else:
                    ax_loss.set_title(f"Retraining: trial {plot_index+1}", fontsize=8, fontweight='bold')

            ax_loss.legend(fontsize=7)
            ax_loss.grid()
            ax_loss.set_yscale('log')

        if self.tr_accuracy != None and fig_acc:
            ax_acc = fig_acc.add_subplot(rows, cols, plot_index + 1)
            ax_acc.plot(self.tr_accuracy, c='r', linestyle='-', label='Training')

            if self.vl_accuracy != None:
                ax_acc.plot(self.vl_accuracy, c='b', linestyle='--', label='Validation')

            if changing_hyperpar:
                for k, v in changing_hyperpar.items():
                    param_name = k.split(".")[-1]
                    ax_acc.plot([], [], " ", label=f"{param_name}: {v}")

            if plot_index >= rows * (cols - 1):
                ax_acc.set_xlabel("Epochs")
            if plot_index % cols == 0:
                ax_acc.set_ylabel("Accuracy / Validation accuracy" if self.ts_accuracy else "Accuracy")

            if self.vl_accuracy is not None:
                if title == "best_model":
                    ax_acc.set_title(f"Best model (ACC: {self.vl_accuracy[-1]:.2%})", fontsize=8, fontweight='bold')
                else:
                    ax_acc.set_title(f"Trial {plot_index+1} (VL: {self.vl_accuracy[-1]:.2%})", fontsize=8, fontweight='bold')
            else:
                if title == "best_model":
                    ax_acc.set_title(f"Retraining", fontsize=8, fontweight='bold')
                else:
                    ax_acc.set_title(f"Retraining: trial {plot_index+1}", fontsize=8, fontweight='bold')

            ax_acc.legend(fontsize=7)
            ax_acc.grid()

        if plot_index == rows * cols - 1:
            if fig_loss is not None:
                fig_loss.subplots_adjust(hspace=0.5)
                fig_loss.savefig(f'../../plots/{title}_loss.png', dpi=300)
            if fig_acc is not None:
                fig_acc.subplots_adjust(hspace=0.5)
                fig_acc.savefig(f'../../plots/{title}_accuracy.png', dpi=300)
            plt.tight_layout()
            plt.show()
        

