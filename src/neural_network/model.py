import activations as actfun
import losses

import numpy as np
import matplotlib.pyplot as plt

import pickle
import copy
import datetime


class Neuron:
    """
    Represents a single neuron in the neural network.

    Attributes
    ----------
    bias : float or None
        Bias value of the neuron
    weights : list of float
        List of weight values of the neuron (one for all other neurons connected)
    """
    def __init__(self):
        self.bias = None
        self.weights = []
    
class NeuronLayer:
    """
    Represents a (dense) layer of neurons in the neural network.
    
    Parameters
    ----------
    neurons : int
        neurons of the layer
        
    Attributes
    ----------
    neurons : list of Neurons
        List of Neurons in the layer
    num_neurons : int
        Number of neurons in the layer
    weights : np.ndarray or None
        Values of weights of all neurons in the layer
    biases : np.ndarray or None
        Value of biases of all neurons in the layer
    inputs : np.ndarray or None
        Inputs of the layer from feed forward
    net : np.ndarray or None

    outputs : np.ndarray or None
        Outputs of the layer (after activation)
    bp_deltas : np.ndarray or None
        Deltas of the backpropagation
    delta_weights_old : np.ndarray or None
        Deltas of the weights of the previous epoch
    delta_biases_old : np.ndarray or None
        Deltas of the biases of the previous epoch
    weights_grad_acc : np.ndarray or None
        Gradient accumulator for weights
    biases_grad_acc : np.ndarray or None
        Gradient accumulator for biases
    """
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
    """
    Represents a multi-layer perceptron neural network for regression or classification.
    
    Parameters
    ----------
    num_inputs : int
        Number of input units
    num_outputs : int
        Number of output units
    neurons_per_layer : list of int
        Number of units for each layers
    training_hyperpar : dict, optional
        All the hyperparameter (epochs, learning_rate, momentum, regularization, batch, early_stopping, etc.)
    extractor : callable, optional
        Function for extract initial weights and biases
    activation : list of list, optional
        Activation functions for hidden and output layers (with coefficients)
    preprocessing : list, optional
        Type of preprocessing of dataset
    
    Attributes
    ----------
    num_inputs : int
        Number of input units
    num_outputs : int
        Number of output units
    neurons_per_layer : list of int
        Number of units for each layers
    hidden_layers_number : int
        Number of hidden layers
    hidden_layers : list of NeuronLayer
        List of all hidden layers
    output_layer : NeuronLayer
        Output layer
    layers : list of NeuronLayer
        List of all layers (hidden + output).
    learning_rate : float
        Current learning rate (eta)
    orig_learning_rate : float
        Initial learning rate (eta_0)
    momentum : float
        Momentum coefficent (alpha)
    nesterov : bool
        Choice for use of Nesterov momentum
    regularization : float
        Regularization (L2) coefficent (lambda)
    preprocessing : str or None
        Type of preprocessing ('standardization', 'rescaling' or None)
    X_params : tuple or None
        Patterns preprocessing parameters (mean/std or min/max)
    T_params : tuple or None
        Targets preprocessing parameters (mean/std or min/max)
    hidden_activation : callable
        Activation function for hidden layers
    output_activation : callable
        Activation function for output layer
    loss_func : callable or None
        Loss function
    tr_loss : list of float or None
        All loss computed on training set
    tr_accuracy : list of float or None
        All accuracy computed on training set
    vl_loss : list of float or None
        All loss computed on validation set
    vl_accuracy : list of float or None
        All accuracy computed on validation set
    best_epoch : int or None
        Epoch of the best model from early stopping
    best_loss : float or None
        Best lost computed
    best_accuracy : float or None
        Best accuracy computed
    
    Methods
    -------
    init_weights(layer, num_inputs, num_outputs)
        Initialize weights and biases
    feed_forward(inputs)
        Do the feed forward of inputs through the network
    back_prop(target)
        Do the backpropagation
    weights_update(batch_size)
        Updates weights and biases with computed gradients
    train(X_tr, T_tr, X_vl, T_vl, train_args, loss_func, early_stopping)
        Train the network
    predict(X)
        Do predictions of new patterns
    loss_calculator(X, T)
        Compute the loss
    accuracy_calculator(X, T)
        Compute the accuracy (only for classification)
    save_model(filepath)
        Save the model on .pkl file
    plot_metrics(fig_loss, fig_acc, rows, cols, plot_index, num_trials, changing_hyperpar, title, data_type)
        Plot the metrics of the network
    """
    def __init__(self, num_inputs, num_outputs, neurons_per_layer, training_hyperpar=None, extractor=None, activation=[["relu", 0.], ["sigmoid", 1.]], preprocessing=[None, None, None]):

        # saving data preprocessing
        self.preprocessing = preprocessing[0]
        if self.preprocessing is not None:
            self.X_params = preprocessing[1]
            self.T_params = preprocessing[2]
        else:
            self.X_params = None
            self.T_params = None

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

        self.hidden_layers = []
        self.output_layer = []

        # initialize hidden neuron and weights in a loop
        for i in range(self.hidden_layers_number):
            num_neurons = self.neurons_per_layer[i]
            #only hidden weights number can be chosen
            num_inputs = self.num_inputs if i == 0 else self.neurons_per_layer[i-1]

            hidden_layer = NeuronLayer(num_neurons)
            if self.extractor is not None:
                self.init_weights(hidden_layer, num_inputs, self.neurons_per_layer[i])
                hidden_layer.weights = np.array([neuron.weights for neuron in hidden_layer.neurons])
                hidden_layer.biases = np.array([neuron.bias for neuron in hidden_layer.neurons])
            else:
                hidden_layer.weights = np.empty((num_neurons, num_inputs))
                hidden_layer.biases = np.empty(num_neurons)

            self.hidden_layers.append(hidden_layer)

        # initialize output neuron and weights
        self.output_layer = NeuronLayer(num_outputs)
        if self.extractor is not None:
            self.init_weights(self.output_layer, self.hidden_layers[-1].num_neurons, num_outputs)
            self.output_layer.weights = np.array([neuron.weights for neuron in self.output_layer.neurons])
            self.output_layer.biases = np.array([neuron.bias for neuron in self.output_layer.neurons])
        else:
            last_hidden_size = self.hidden_layers[-1].num_neurons if self.hidden_layers else self.num_inputs
            self.output_layer.weights = np.empty((num_outputs, last_hidden_size))
            self.output_layer.biases = np.empty(num_outputs)

        self.layers = self.hidden_layers + [self.output_layer]

        for layer in self.layers:
            layer.delta_weights_old = np.zeros_like(layer.weights)
            layer.delta_biases_old = np.zeros_like(layer.biases)
        

        if training_hyperpar is not None:
            self.orig_learning_rate = training_hyperpar["learning_rate"]["eta"]
            self.learning_rate = self.orig_learning_rate
            self.min_learning_rate = self.orig_learning_rate / training_hyperpar["learning_rate"]["min_rate"]
            self.decay_factor = training_hyperpar["learning_rate"]["decay_factor"]
            self.momentum = training_hyperpar["momentum"]
            self.nesterov = training_hyperpar["nesterov"]
            self.regularization = training_hyperpar["regularization"]

        self.loss_func = None
        self.tr_loss = None
        self.tr_accuracy = None
        self.vl_loss = None
        self.vl_accuracy = None
        self.ts_loss = None
        self.ts_accuracy = None
        self.best_epoch = None
        self.best_loss = None
        self.best_accuracy = None


    def init_weights(self, layer, num_inputs, num_outputs):
        """
        Initialize weights and biases of the layer

        Parameters
        ----------
        layer : NeuronLayer
            Layer of weights and biases to be initialized
        num_inputs : int
            Number of input neurons
        num_outputs : int
            Number of outputs neurons
        """
        for neuron in layer.neurons: #use a function as a sort of random generator to create the values for bias and weights
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
    
    def feed_forward(self, inputs):
        """
        Propagate inputs through the network.

        Parameters
        ----------
        inputs : np.ndarray
            Patterns vector

        Returns
        -------
        np.ndarray
            Network outputs
        """
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
        """
        Backpropagate targets through the inputs.

        Parameters
        ----------
        target : np.ndarray
            Target of the current pattern
        """
        previous_delta = None
        previous_weights = None
        for layer in reversed(self.layers):
            # output layer
            if layer == self.output_layer:
                if self.loss_func.__name__ == "binary_crossentropy" and self.output_activation.__name__ == "sigmoid":
                    layer.bp_deltas = target - layer.outputs
                else:
                    layer.bp_deltas = (target - layer.outputs) * self.d_output_activation(layer.net, self.output_activation_param)                       # delta_k
            # hidden layers
            elif layer in self.hidden_layers:
                layer.bp_deltas = (previous_weights.T @ previous_delta) * self.d_hidden_activation(layer.net, self.hidden_activation_param)          # delta_j
            else:
                raise Exception("Layer not recognized")
            
            previous_delta, previous_weights = layer.bp_deltas, layer.weights
 
    def weights_update(self, batch_size):
        """
        Update of the weights and biases using gradients from backpropagation (delta_w = eta * gradient + alpha * delta_w_old, w_new = w + delta_w - lambda * w)

        Parameters
        ----------
        batch_size : int
            Number of patterns in the (mini)batch
        """
        for layer in self.layers:
            # gradient calculation
            if batch_size == 1:
                grad_w = self.learning_rate * np.outer(layer.bp_deltas, layer.inputs)
                grad_b = self.learning_rate * layer.bp_deltas
            else:
                grad_w = self.learning_rate * layer.weights_grad_acc
                grad_b = self.learning_rate * layer.biases_grad_acc

            # delta_new = gradient + momentum  
            delta_weights = grad_w + self.momentum * layer.delta_weights_old
            delta_biases = grad_b + self.momentum * layer.delta_biases_old
            
            # update new weights
            layer.weights += delta_weights - self.regularization * layer.weights
            layer.biases += delta_biases - self.regularization * layer.biases

            # save current delta
            layer.delta_weights_old = delta_weights
            layer.delta_biases_old = delta_biases

    def train(self, X_tr, T_tr, X_vl=None, T_vl=None, X_ts=None, T_ts=None, train_args=None, loss_func=None, early_stopping=None):
        """
        Train the entire neural network.

        Parameters
        ----------
        X_tr : np.ndarray
            Vector of training input data
        T_tr : np.ndarray
            Vector of training target
        X_vl : np.ndarray, optional
            Vector of validation input data
        T_vl : np.ndarray, optimal
            Vector of validation target
        train_args : dict
            Contain:
            - "batch": "batch_size" (int or "full"), "drop_last" (bool)
            - "epochs" (int) max epoch to reach
        loss_func : str
            Name of the loss function
        early_stopping : dict
            Contain:
            - "enabled" (bool)
            - "patience" (int) epochs to wait before stopping
            - "monitor" (str) metric to monitor ("val_loss", "val_accuracy", "train_loss")
            - "epsilon_loss_down" (float) loss relative improvement threshold
            - "epsilon_accuracy_up" (float) accuracy relative improvement threshold
            - "target_loss" (float) target loss for "train_loss" monitoring (used only for retraining)
        
        Raises
        ------
        ValueError
            If monitor parameter is not valid or target_loss is missing when needed
        TypeError
            If batch_size is not a positive integer or "full"
        """
        batch_size = train_args["batch"]["batch_size"]
        batch_droplast = train_args["batch"]["drop_last"]
        max_epochs = train_args["epochs"]

        # early stopping variables
        early_stopping_cond = early_stopping["enabled"]
        patience = early_stopping["patience"] if early_stopping_cond else None
        monitor = early_stopping["monitor"] if early_stopping_cond else None
        target_loss = early_stopping["target_loss"] if early_stopping_cond else None
        
        if early_stopping_cond:    
            if monitor == "val_loss":
                epsilon_down = early_stopping["epsilon_loss_down"]
            elif monitor == "val_accuracy":
                epsilon_up = early_stopping["epsilon_accuracy_up"]
            elif monitor == "train_loss":
                if target_loss is None:
                    raise ValueError('target_loss must be provided when monitor is "train_loss"')
            else:
                raise ValueError('monitor parameter must be "val_loss", "val_accuracy" or "train_loss"')

        if X_vl is not None:
            best_model_weights = copy.deepcopy(self.layers)

        self.loss_func = losses.losses_functions[loss_func]
        tr_loss = []
        tr_accuracy = []
        vl_loss = []
        vl_accuracy = []
        ts_loss = []
        ts_accuracy = []

        tr_loss.append(self.loss_calculator(X_tr, T_tr))                 # loss 0: with random parameters
        tr_accuracy.append(self.accuracy_calculator(X_tr, T_tr))

        best_loss = None
        best_accuracy = None

        if X_vl is not None:        #get loss for epoch 0 on vl set
            init_vl_loss = self.loss_calculator(X_vl, T_vl)
            init_vl_accuracy = self.accuracy_calculator(X_vl, T_vl)

            best_loss = init_vl_loss
            best_accuracy = init_vl_accuracy

            vl_loss.append(init_vl_loss) 
            vl_accuracy.append(init_vl_accuracy)

        if X_ts is not None:        #get loss for epoch 0 on ts set
            init_ts_loss = self.loss_calculator(X_ts, T_ts)
            init_ts_accuracy = self.accuracy_calculator(X_ts, T_ts)

            ts_loss.append(init_ts_loss) 
            ts_accuracy.append(init_ts_accuracy)

        current_epoch = 0
        patience_index = patience if patience is not None else float('inf')
        best_epoch = 0
        
        while current_epoch < max_epochs and (patience_index > 0 if early_stopping_cond else True):
            if self.decay_factor > 0 and self.learning_rate > self.min_learning_rate:
                self.learning_rate = self.orig_learning_rate / (1 + self.decay_factor * current_epoch)
            current_epoch += 1
            
            if batch_size == "full":
                #for batch gradient descent, i need to have a way to accumulate the gradient for each pattern with a shape like the weights
                for layer in self.layers:
                    layer.weights_grad_acc = np.zeros_like(layer.weights)
                    layer.biases_grad_acc = np.zeros_like(layer.biases)
            
                if self.nesterov:
                    real_weights = [layer.weights.copy() for layer in self.layers]
                    real_biases = [layer.biases.copy() for layer in self.layers]
                    
                    for layer in self.layers:
                        layer.weights += self.momentum * layer.delta_weights_old
                        layer.biases += self.momentum * layer.delta_biases_old

                #iterate on each pattern of and backprop
                for x,t in zip(X_tr, T_tr):
                    self.feed_forward(x)
                    self.back_prop(t)

                    #accumulate gradient
                    for layer in self.layers:
                        layer.weights_grad_acc += np.outer(layer.bp_deltas, layer.inputs)
                        layer.biases_grad_acc += layer.bp_deltas
                
                if self.nesterov:
                    for layer, real_w, real_b in zip(self.layers, real_weights, real_biases):
                        layer.weights = real_w
                        layer.biases = real_b

                self.weights_update(len(X_tr))

            elif isinstance(batch_size, int) and batch_size > 0:        #manage mini batch and sgd cases
                perm = np.random.permutation(len(X_tr))
                X_tr = X_tr[perm]
                T_tr = T_tr[perm]
                
                if batch_size != 1:
                    for layer in self.layers: 
                        layer.weights_grad_acc = np.zeros_like(layer.weights)
                        layer.biases_grad_acc = np.zeros_like(layer.biases)
                    
                    counter = 0
                    for x,t in zip(X_tr,T_tr):
                        if counter == 0 and self.nesterov:
                            real_weights = [layer.weights.copy() for layer in self.layers]
                            real_biases = [layer.biases.copy() for layer in self.layers]
                            for layer in self.layers:
                                layer.weights += self.momentum * layer.delta_weights_old
                                layer.biases += self.momentum * layer.delta_biases_old

                        self.feed_forward(x)
                        self.back_prop(t)

                        #using a counter manages the istances if dataset ends before batch size is reached
                        counter += 1

                        for layer in self.layers:
                            layer.weights_grad_acc += np.outer(layer.bp_deltas, layer.inputs)
                            layer.biases_grad_acc += layer.bp_deltas 
                        
                        if counter == batch_size:
                            if self.nesterov:
                                for layer, real_w, real_b in zip(self.layers, real_weights, real_biases):
                                    layer.weights = real_w
                                    layer.biases = real_b

                            self.weights_update(counter)
                            for layer in self.layers: 
                                layer.weights_grad_acc = np.zeros_like(layer.weights)
                                layer.biases_grad_acc = np.zeros_like(layer.biases)
                            counter = 0 

                    #flush update if necessary
                    if counter != 0 and not batch_droplast:
                        if self.nesterov:
                            for layer, real_w, real_b in zip(self.layers, real_weights, real_biases):
                                layer.weights = real_w
                                layer.biases = real_b
                        
                        self.weights_update(counter)

                elif batch_size == 1: 
                    for x,t in zip(X_tr,T_tr):
                        if self.nesterov:
                            real_weights = [layer.weights.copy() for layer in self.layers]
                            real_biases = [layer.biases.copy() for layer in self.layers]
                            for layer in self.layers:
                                layer.weights += self.momentum * layer.delta_weights_old
                                layer.biases += self.momentum * layer.delta_biases_old

                        self.feed_forward(x)
                        self.back_prop(t)

                        if self.nesterov:
                            for layer, real_w, real_b in zip(self.layers, real_weights, real_biases):
                                layer.weights = real_w
                                layer.biases = real_b

                        self.weights_update(1)

            else:
                raise TypeError('batch_size is not positive int or "full".')

            current_tr_loss = self.loss_calculator(X_tr, T_tr)
            current_tr_accuracy = self.accuracy_calculator(X_tr, T_tr)
            
            if X_vl is not None:        #get loss for current epoch on vl set
                current_vl_loss = self.loss_calculator(X_vl, T_vl)
                vl_loss.append(current_vl_loss)
                current_vl_accuracy = self.accuracy_calculator(X_vl, T_vl)
                vl_accuracy.append(current_vl_accuracy)
            
            if X_ts is not None:        #get loss for current epoch on vl set
                current_ts_loss = self.loss_calculator(X_ts, T_ts)
                ts_loss.append(current_ts_loss)
                current_ts_accuracy = self.accuracy_calculator(X_ts, T_ts)
                ts_accuracy.append(current_ts_accuracy)
            
            #check if vl increases
            if early_stopping_cond and monitor == "val_loss":

                if current_vl_loss <= np.min(vl_loss[:-1]) * (1 - epsilon_down):
                    patience_index = patience
                    best_model_weights = copy.deepcopy(self.layers)
                    best_epoch = current_epoch
                    best_loss = current_vl_loss
                    best_accuracy = current_vl_accuracy
                else:
                    patience_index -= 1

            elif early_stopping_cond and monitor == "val_accuracy":
                if current_vl_accuracy >= np.max(vl_accuracy[:-1]) * (1 + epsilon_up):
                    patience_index = patience
                    best_model_weights = copy.deepcopy(self.layers)
                    best_epoch = current_epoch
                    best_loss = current_vl_loss
                    best_accuracy = current_vl_accuracy
                else:
                    patience_index -= 1
                
            elif early_stopping_cond and monitor == "train_loss":
                if current_tr_loss < target_loss:
                    patience_index = 0
                    best_epoch = current_epoch
                    best_loss = current_tr_loss
                    best_accuracy = current_tr_accuracy
                    if X_vl is None:
                        best_model_weights = copy.deepcopy(self.layers)

            tr_loss.append(current_tr_loss)
            tr_accuracy.append(current_tr_accuracy)

        print(f"Early stopping at epoch: {current_epoch}" if (early_stopping_cond and patience_index == 0) else f"Max epoch reached ({max_epochs})")

        #assign the best values
        if X_vl is not None:
            self.layers = copy.deepcopy(best_model_weights)

        if X_vl is not None:
            self.best_epoch = best_epoch
            self.best_loss = best_loss
            self.best_accuracy = best_accuracy
        else:
            self.best_epoch = current_epoch
            self.best_loss = current_tr_loss
            self.best_accuracy = current_tr_accuracy

        self.hidden_layers = self.layers[:-1]
        self.output_layer = self.layers[-1]

        self.tr_loss = tr_loss
        self.tr_accuracy = tr_accuracy
        if X_vl is not None:
            self.vl_loss = vl_loss
            self.vl_accuracy = vl_accuracy
        if X_ts is not None:
            self.ts_loss = ts_loss
            self.ts_accuracy = ts_accuracy
    
    def loss_calculator(self, X, T):
        """
        Compute loss

        Parameters
        ----------
        X : np.ndarray
            Vector of input data
        T : np.ndarray
            Vector of target data

        Returns
        -------
        float
            Loss of all patterns
        """
        predictions = np.array([self.feed_forward(x) for x in X])
        loss = self.loss_func(predictions, T)
        return loss
    
    def accuracy_calculator(self, X, T):
        """
        Compute accuracy (only for classification)

        Parameters
        ----------
        X : np.ndarray
            Vector of input data
        T : np.ndarray
            Vector of target data

        Returns
        -------
        float or None
            Accuracy of all patterns (from 0 to 1). It is None if problem is not a classification
        """
        if self.output_activation.__name__ not in ["sigmoid", "tanh"]:
            return None
        
        correct_predict = 0
        threshold = 0.5 if self.output_activation.__name__ == "sigmoid" else 0.

        for x,t in zip(X,T):
            predictions = self.feed_forward(x)

            if predictions.shape[0] == 1:
                if (predictions >= threshold and t == 1) or (predictions < threshold and t == 0):
                    correct_predict += 1
            else:
                if np.all(predictions >= threshold and t == 1) or np.all(predictions < threshold and t == 0):
                    correct_predict += 1

        accuracy = correct_predict / len(T)
        return accuracy

    # def test(self, X, T):
    #     correct_predict = 0
    #     for x,t in zip(X,T):
    #         o = self.feed_forward(x)
    #         if o >= 0.5 and t == 1 or o < 0.5 and t == 0:
    #             correct_predict += 1
    #     accuracy = correct_predict/len(T)
    #     print(f"The model obtained an accuracy of {accuracy:.2%} on test set")

    def predict(self, X):
        """
        Predict input data

        Parameters
        ----------
        X : np.ndarray
            Vector of input data
        
        Returns
        -------
        np.ndarray
            Vector of predicted output data
        """
        O = []
        for x in X:
            o = self.feed_forward(x)
            O.append(o)
        return np.array(O)
    
    def plot_metrics(self, fig_loss=None, fig_acc=None, rows=1, cols=1, plot_index=0, num_trials=None, changing_hyperpar=None, title=None, data_type=None, save_path=None):
        """
        Plot the metrics of the network

        Parameters
        ----------
        fig_loss : plt.figure.Figure
            Figure of loss plot
        fig_acc : plt.figure.Figure, optional
            Figure of accuracy plot
        rows : int, optional
            Number of rows for subplots
        cols : int, optional
            Number of cols for subplots
        plot_index : int, optional
            Index of the current subplot (usefull for grid_search and random_search)
        num_trials : int, optional
            Number of total trials for saveing figures
        changing_hyperpar : dict, optional
            Dictionary of hyperparameters that changed for this trial
        title : str, optional
            Title of the plot ("single_trial", "best_model", "grid_search", "random_search")
        data_type : str, optional
            Dataset name ("CUP", "MONK")
        save_path : str, optional
            Path for saving plots
        """
        if self.tr_loss is not None and fig_loss:
            ax_loss = fig_loss.add_subplot(rows, cols, plot_index + 1)
            ax_loss.plot(self.tr_loss, c='r', linestyle='-', label='Training')

            if self.vl_loss is not None:
                ax_loss.plot(self.vl_loss, c='b', linestyle='--', label='Validation')
            elif self.ts_loss is not None:
                ax_loss.plot(self.ts_loss, c='b', linestyle='--', label='Test')

            if changing_hyperpar:
                for k, v in changing_hyperpar.items():
                    param_name = k.split(".")[-1]
                    ax_loss.plot([], [], " ", label=f"{param_name}: {v}")

            if plot_index >= rows * (cols - 1):
                ax_loss.set_xlabel("Epochs")
            if plot_index % cols == 0:
                ax_loss.set_ylabel("Loss / Validation loss" if self.vl_loss is not None else "Loss / Risk" if self.ts_loss is not None else "Loss")

            if self.best_loss is not None:
                if title == "best_model":
                    ax_loss.set_title(f"Best model (VL: {self.best_loss:.4f})", fontsize=14, fontweight='bold')
                elif title == "single_trial":
                    ax_loss.set_title(f"Single trial (VL: {self.best_loss:.4f})", fontsize=14, fontweight='bold')
                else:
                    ax_loss.set_title(f"Trial {plot_index+1} (VL: {self.best_loss:.4f})", fontsize=14, fontweight='bold')
            else:
                if title == "best_model":
                    ax_loss.set_title(f"Retraining", fontsize=14, fontweight='bold')
                else:
                    ax_loss.set_title(f"Retraining: trial {plot_index+1}", fontsize=14, fontweight='bold')

            ax_loss.legend(fontsize=12)
            ax_loss.grid()
            ax_loss.set_yscale('log')

        if self.tr_accuracy is not None and fig_acc:
            ax_acc = fig_acc.add_subplot(rows, cols, plot_index + 1)
            ax_acc.plot(self.tr_accuracy, c='r', linestyle='-', label='Training')

            if self.vl_accuracy is not None:
                ax_acc.plot(self.vl_accuracy, c='b', linestyle='--', label='Validation')
            elif self.ts_accuracy is not None:
                ax_acc.plot(self.ts_accuracy, c='b', linestyle='--', label='Test')

            if self.vl_accuracy is not None:
                if changing_hyperpar:
                    for k, v in changing_hyperpar.items():
                        param_name = k.split(".")[-1]
                        ax_acc.plot([], [], " ", label=f"{param_name}: {v}")

            if plot_index >= rows * (cols - 1):
                ax_acc.set_xlabel("Epochs")
            if plot_index % cols == 0:
                ax_acc.set_ylabel("Accuracy / Validation accuracy" if self.vl_accuracy is not None else "Accuracy / Risk accuracy" if self.ts_accuracy is not None else "Accuracy")

            if self.output_activation.__name__ in ["sigmoid", "tanh"]:
                if self.best_accuracy is not None:
                    val_acc_text = f"{self.best_accuracy:.2%}"
                else:
                    val_acc_text = "N/A"

                if title == "best_model":
                    ax_acc.set_title(f"Best model (ACC: {val_acc_text})", fontsize=12, fontweight='bold')
                elif title == "single_trial":
                    ax_acc.set_title(f"Single trial (ACC: {val_acc_text})", fontsize=12, fontweight='bold')
                else:
                    ax_acc.set_title(f"Trial {plot_index+1} (VL: {val_acc_text})", fontsize=12, fontweight='bold')
            else:
                if title == "best_model":
                    ax_acc.set_title(f"Retraining", fontsize=12, fontweight='bold')
                else:
                    ax_acc.set_title(f"Retraining: trial {plot_index+1}", fontsize=12, fontweight='bold')

            ax_acc.legend(fontsize=12)
            ax_acc.grid()

        if plot_index == (num_trials - 1 if num_trials is not None else rows * cols - 1):
            if title in ["grid_search", "random_search"]:
                fig_loss.subplots_adjust(hspace=0.5)
                if self.output_activation.__name__ in ["sigmoid", "tanh"]:
                    fig_acc.subplots_adjust(hspace=0.5)
            else:
                if fig_loss is not None:
                    fig_loss.tight_layout()
                    if save_path is not None:
                        fig_loss.savefig(f'{save_path}/{data_type}_{title}_loss.png', dpi=300)
                    else:
                        fig_loss.savefig(f'../../plots/{data_type}_{title}_loss.png', dpi=300)
                    plt.close(fig_loss)
                if self.output_activation.__name__ in ["sigmoid", "tanh"]:
                    fig_acc.tight_layout()
                    if save_path is not None:
                        fig_acc.savefig(f'{save_path}/{data_type}_{title}_accuracy.png', dpi=300)
                    else:
                        fig_acc.savefig(f'../../plots/{data_type}_{title}_accuracy.png', dpi=300)
                    plt.close(fig_acc)
            # plt.show()

    def save_model(self, filepath):
        """
        Save the model on .pkl file

        Parameters
        ----------
        filepath : str
            Path to save the file
        """
        self.sync_layer_to_neurons()
        
        model_config = {
            "architecture": {
                "input_units": self.num_inputs,
                "output_units": self.num_outputs,
                "neurons_per_layer": self.neurons_per_layer
            },
            "functions": {
                "hidden": self.hidden_activation.__name__,
                "hidden_param": self.hidden_activation_param,
                "output": self.output_activation.__name__,
                "output_param": self.output_activation_param
            },
            "preprocessing": {
                "type": self.preprocessing,
                "X_params": self.X_params,
                "T_params": self.T_params
            },
            "layers": [{
                "weights": layer.weights,
                "biases": layer.biases
            } for layer in self.layers],
            "training_hyperpar": {
                "learning_rate": self.orig_learning_rate,
                "momentum": self.momentum,
                "regularization": self.regularization
            },
            "date": datetime.datetime.now().isoformat()
        }

        with open(filepath, "wb") as file:
            pickle.dump(model_config, file)

    def sync_layer_to_neurons(self):
        """
        Synchronize layer weights (and biases) value with neurons 
        """
        for layer in self.layers:
            for i, neuron in enumerate(layer.neurons):
                neuron.weights = layer.weights[i].tolist()
                neuron.bias = float(layer.biases[i])
