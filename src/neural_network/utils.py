from model import NeuralNetwork

import numpy as np
import pandas as pd

import pickle
import json
import datetime


def load_config_json(filepath):
    """Loads the configuration from the specified JSON file."""
    with open(filepath, 'r') as f:
        config = json.load(f)
    return config

#can choose between different extraction methods for the weights/biases
def create_extractor(method, range):
    if(method == "random"):
        def random_extractor(fan_in=None, fan_out=None, a=None):
            return np.random.uniform(-range/2, range/2)
        return random_extractor
    elif(method == "fan_in"):   # Micheli (Slide NN-part2 n.12) 
        def fanin_extractor(fan_in=None, fan_out=None, a=None):
            bound = range * 2. / fan_in
            return np.random.uniform(-bound, bound)
        return fanin_extractor
    elif(method == "xavier"):   # sigmoid/tanh
        def xavier_extractor(fan_in=None, fan_out=None, a=None):
            bound = np.sqrt(6. / (fan_in + fan_out))
            return np.random.uniform(-bound, bound)
        return xavier_extractor
    elif(method == "he"):   # ReLu
        def he_extractor(fan_in=None, fan_out=None, a=None):
            bound = np.sqrt(6. / (fan_in * (1 + a**2)))
            return np.random.uniform(-bound, bound)
        return he_extractor
    else:
        raise ValueError("Invalid method.")
    
def standardization(X):
    eps = 1e-8                              # for non zero std
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + eps
    return mean, std
    
def scaling(X):                        # remember to do the inverse at the end with "inverse_scaling"
    eps = 1e-8                              # for non zero deltaX
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0) + eps
    return X_min, X_max

def inverse_standardization(X, X_mean, X_std):
    return X * X_std + X_mean            #returns un-standardized results

def inverse_scaling(X, X_min, X_max):
    return X * (X_max - X_min) + X_min           #returns un-scaled results

def data_splitting(X, T, proportions=[1,0,0]):
    if any(prop < 0 for prop in proportions):
        raise ValueError('Elements in "proportions" must be greater or equal than 0')
    
    if sum(proportions) != 1:
        proportions = [i/sum(proportions) for i in proportions]

    n = len(X)
    X_splits = []
    T_splits = []

    start = 0
    for p in proportions:
        end = start + int(p * n)

        X_splits.append(X[start:end])
        T_splits.append(T[start:end])
        
        start = end

    # returning results as separate vectors
    return (*X_splits, *T_splits)


def flatten_config(d, parent_key='', sep='.'):
    '''
    Loads the json in memory keeping the same structure as the json but with multiple values
    '''
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k          #used to create a trail when iterating through nested elements of the config json
        
        
        if isinstance(v, dict):         #if there is a dictionary, recursively call the flattening and add the dictionary name as parent key
            items.extend(flatten_config(v, new_key, sep=sep).items())

        elif isinstance(v, list):
            items.append((new_key, v))      #otherwise, just add the leaf
            
        else:
            items.append((new_key, [v]))
            
    return dict(items)

def set_dict(d, k, v, sep="."):
    keys = k.split(sep)
    for key in keys[:-1]:
        d = d.setdefault(key, {})

    d[keys[-1]] = v

def loguniform(v0, v1):
    return 10 ** np.random.uniform(np.log10(v0), np.log10(v1))

def get_k_fold_indices(n_samples, k_folds):
    folds_size = int(np.floor(n_samples / k_folds))
    folds_remainder = n_samples % k_folds
    
    folds_indexes = []
    start_index = 0
    
    for i in range(k_folds):            #calculates the start and ending indexes for each fold
        current_fold_size = folds_size + (1 if i < folds_remainder else 0)
        current_end = start_index + current_fold_size
        folds_indexes.append((start_index, current_end))
        start_index = current_end
        
    return folds_indexes


def neural_network_from_file(file_path):
    with open(file_path, "rb") as file:
        config = pickle.load(file)

    hidden_act_func = [config["functions"]["hidden"], config["functions"]["hidden_param"]]
    output_act_func = [config["functions"]["output"], config["functions"]["output_param"]]
    act_func = [hidden_act_func, output_act_func]

    preprocessing = [config["preprocessing"]["type"], config["preprocessing"]["X_params"], config["preprocessing"]["T_params"]]

    nn = NeuralNetwork(num_inputs=config["architecture"]["input_units"], num_outputs=config["architecture"]["output_units"], neurons_per_layer=config["architecture"]["neurons_per_layer"], activation=act_func, preprocessing=preprocessing)

    for layer, config_layer in zip(nn.layers, config["layers"]):
        layer.weights = config_layer["weights"]
        layer.biases = config_layer["biases"]

        for i, neuron in enumerate(layer.neurons):
            neuron.weights = config_layer["weights"][i].tolist()
            neuron.bias = float(config_layer["biases"][i])
    
    nn.hidden_layers = nn.layers[:-1]
    nn.output_layer = nn.layers[-1]
                
    return nn

def save_predictions(filename, predictions, team_name, members):
    date = datetime.datetime.now().strftime("%d/%m/%Y")

    df = pd.DataFrame(predictions)
    df.index += 1
    df.reset_index(inplace=True)

    with open(filename, 'w', newline='') as f:
        f.write(f"# {members}\n")
        f.write(f"# {team_name}\n")
        f.write("# ML-CUP25\n")
        f.write(f"# {date}\n")
        df.to_csv(f, index=False, header=False)

