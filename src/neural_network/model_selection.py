import utils
import activations as actfun
import losses
import itertools
from model import NeuralNetwork

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def grid_search(training_sets, input_units, config):

    flattened_values = utils.flatten_config(config)

    keys, values = zip(*flattened_values.items())

    trials = []
    for comb in itertools.product(*values):
        new_config = {}

        for k, v in zip(keys, comb):
            utils.set_dict(new_config, k, v)
        trials.append(new_config)

    # identify which key changes
    flattened_trials = [utils.flatten_config(trial) for trial in trials]
    all_keys = flattened_trials[0].keys()
    
    changing_keys = []
    for key in all_keys:
        values_for_key = [trial[key] for trial in flattened_trials]
        unique_values = set()
        for v in values_for_key:
            unique_values.add(tuple(v))
        
        if len(unique_values) > 1:
            changing_keys.append(key)
    
    # identify which parameter changes
    changing_hyperpar = []
    for trial in flattened_trials:
        trial_changing = {k: trial[k] for k in changing_keys}
        changing_hyperpar.append(trial_changing)

    num_trials = len(trials)
    n_cols = int(np.ceil(np.sqrt(num_trials)))
    n_rows = int(np.ceil(num_trials / n_cols))
    
    fig_loss = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    fig_acc = plt.figure(figsize=(5 * n_cols, 4 * n_rows))

    best_vl_loss = []
    networks_tried = []
    for i, trial in enumerate(trials):
        loss, nn = launch_trial(trial, training_sets, input_units)
        networks_tried.append((loss, nn))
        
        best_vl_loss = [loss, i] if not best_vl_loss or loss < best_vl_loss[0] else best_vl_loss

        nn.plot_metrics(fig_loss=fig_loss, fig_acc=fig_acc, rows=n_rows, cols=n_cols, plot_index=i, changing_hyperpar=changing_hyperpar[i])

    print(f"The best combination is:\n{trials[best_vl_loss[1]]}\n\nwith a vl loss of {best_vl_loss[0]}\n\n\n")
    #TODO ricordarsi di rifare training del modello con parametri ottimi dopo la vl

def random_search(training_sets, input_units, config):

    flattened_values = utils.flatten_config(config)

    keys, values = zip(*flattened_values.items())

    hyperpar_error = []
    hyperpar_bounded = []
    hyperpar_constant = []
    for idx, val in enumerate(values):
        if len(val) == 1:
            hyperpar_constant.append(keys[idx])
        elif len(val) == 2:
            hyperpar_bounded.append(keys[idx])
        elif len(val) > 2:
            hyperpar_error.append(keys[idx])
    if len(hyperpar_error) != 0:
        raise ValueError(f'The hyperparameter bounds must be 2 for random search. The hyperparameter(s) that create problem are:{hyperpar_error}')


    trials = []
    num_trials = config["training"]["number_random_trials"]
    for _ in range(num_trials):
        new_config = {}
        
        for k, v in zip(keys, values):

            if k in hyperpar_constant:
                random_value = v[0]

            if k in hyperpar_bounded:

                if k == "momentum":
                    random_value = utils.loguniform(v[0], v[1])

                elif isinstance(v[0], int) and isinstance(v[1], int):
                    random_value = np.random.randint(v[0], v[1] + 1)

                elif isinstance(v[0], float) or isinstance(v[1], float):
                    random_value = np.round(np.random.uniform(v[0], v[1]), 4)                           # TODO rivedere il 4 (scelto arbitrariamente)

                elif isinstance(v[0], list) and isinstance(v[0][0], int):
                    random_value = [np.random.randint(v[0][i], v[1][i] + 1) for i in range(len(v[0]))]
                
                else:
                    raise TypeError(f"Unexpected type: {k}")
                
            utils.set_dict(new_config, k, random_value)
        
        trials.append(new_config)

    # identify which key changes
    flattened_trials = [utils.flatten_config(trial) for trial in trials]
    all_keys = flattened_trials[0].keys()
    
    changing_keys = []
    for key in all_keys:
        values_for_key = [trial[key] for trial in flattened_trials]
        unique_values = set()
        for v in values_for_key:
            unique_values.add(tuple(v))
        
        if len(unique_values) > 1:
            changing_keys.append(key)
    
    # identify which parameter changes
    changing_hyperpar = []
    for trial in flattened_trials:
        trial_changing = {k: trial[k] for k in changing_keys}
        changing_hyperpar.append(trial_changing)

    n_cols = int(np.ceil(np.sqrt(num_trials)))
    n_rows = int(np.ceil(num_trials / n_cols))
    
    fig_loss = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    fig_acc = plt.figure(figsize=(5 * n_cols, 4 * n_rows))

    best_vl_loss = []
    networks_tried = []
    for i, trial in enumerate(trials):
        loss, nn = launch_trial(trial, training_sets, input_units)
        networks_tried.append((loss, nn))
        
        best_vl_loss = [loss, i] if not best_vl_loss or loss < best_vl_loss[0] else best_vl_loss

        nn.plot_metrics(fig_loss=fig_loss, fig_acc=fig_acc, rows=n_rows, cols=n_cols, plot_index=i, changing_hyperpar=changing_hyperpar[i])

    print(f"The best combination is:\n{trials[best_vl_loss[1]]}\n\nwith a vl loss of {best_vl_loss[0]}\n\n\n")
    #TODO ricordarsi di rifare training del modello con parametri ottimi dopo la vl

def launch_trial(comb, training_sets, input_units):
    print(f"Parameters:\n{comb}\n")

    X_train, X_val, T_train, T_val = training_sets

    output_units = comb["architecture"]["output_units"]
    neurons_per_layer = comb["architecture"]["neurons_per_layer"]
    hidden_act_func = comb["functions"]["hidden"]
    output_act_func = comb["functions"]["output"]
    act_func = [hidden_act_func, output_act_func]   

    train_args = comb["training"]
    training_hyperpar = comb["training"]

    early_stopping = comb["training"]["early_stopping"]

    loss_func = comb["functions"]["loss"]

    extractor = utils.create_random_extractor(comb["initialization"]["method"])

    nn = NeuralNetwork(num_inputs=input_units,
                       num_outputs=output_units,
                       neurons_per_layer=neurons_per_layer,
                       training_hyperpar=training_hyperpar,
                       extractor=extractor,
                       activation=act_func,
                       early_stopping=early_stopping)
    
    nn.train(X_train, T_train, X_val, T_val, train_args=train_args, loss_func=loss_func, early_stopping=early_stopping)

    best_vl_loss = nn.loss_calculator(X_val, T_val, losses.losses_functions[loss_func])
    print(f"Best validation loss for this run: {best_vl_loss:.6f}\n")

    return best_vl_loss, nn


def perform_search(training_sets, input_units, config):
    search_type = config["training"]["search_type"]
    if search_type == "grid":
        return grid_search(training_sets, input_units, config)
    if search_type == "random":
        return random_search(training_sets, input_units, config)
    else:
        raise ValueError('"search_type" must be "grid" or "random"')