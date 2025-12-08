import utils
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

    # num_trials = len(trials)
    # n_cols = int(np.round(np.sqrt(num_trials)))
    # n_rows = int(np.round(num_trials / n_cols))
    
    # fig_loss = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    # fig_acc = plt.figure(figsize=(5 * n_cols, 4 * n_rows))

    # run all the possible trial
    best_vl_loss = None
    best_trial_idx = None
    # networks_tried = []
    for i, trial in enumerate(trials):
        print(f"Trial {i+1}/{len(trials)} :\n")

        loss = evaluate_configuration(trial, training_sets, input_units)

        if best_vl_loss is None or loss < best_vl_loss:
            best_vl_loss = loss
            best_trial_idx = i
            print(f"New best loss: {best_vl_loss:.6f}\n")

    best_config = trials[best_trial_idx]

    # retrain on model with the best loss
    X_full = np.concatenate((training_sets[0], training_sets[1]))
    T_full = np.concatenate((training_sets[2], training_sets[3]))

    _, final_nn = launch_trial(best_config, [X_full, X_full, T_full, T_full], input_units, verbose=True)

    fig_loss = plt.figure(figsize=(5, 4))
    fig_acc = plt.figure(figsize=(5, 4))
    final_nn.plot_metrics(fig_loss=fig_loss, fig_acc=fig_acc)

    return final_nn, best_config

        # loss, nn = launch_trial(trial, training_sets, input_units)
        # networks_tried.append((loss, nn))
        
        # best_vl_loss = [loss, i] if not best_vl_loss or loss < best_vl_loss[0] else best_vl_loss

        # nn.plot_metrics(fig_loss=fig_loss, fig_acc=fig_acc, rows=n_rows, cols=n_cols, plot_index=i, changing_hyperpar=changing_hyperpar[i])

    # print(f"The best combination is:\n{trials[best_vl_loss[1]]}\n\nwith a vl loss of {best_vl_loss[0]}\n\n\n")

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

    # n_cols = int(np.round(np.sqrt(num_trials)))
    # n_rows = int(np.round(num_trials / n_cols))
    
    # fig_loss = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    # fig_acc = plt.figure(figsize=(5 * n_cols, 4 * n_rows))

    # run all the possible trial
    best_vl_loss = None
    best_trial_idx = None
    # networks_tried = []
    for i, trial in enumerate(trials):
        print(f"Trial {i+1}/{len(trials)} :\n")

        loss = evaluate_configuration(trial, training_sets, input_units)

        if best_vl_loss is None or loss < best_vl_loss:
            best_vl_loss = loss
            best_trial_idx = i
            print(f"New best found! Loss: {best_vl_loss:.6f}")

    best_config = trials[best_trial_idx]

    # retrain on model with the best loss
    X_full = np.concatenate((training_sets[0], training_sets[1]))
    T_full = np.concatenate((training_sets[2], training_sets[3]))

    _, final_nn = launch_trial(best_config, [X_full, X_full, T_full, T_full], input_units, verbose=True)

    fig_loss = plt.figure(figsize=(5, 4))
    fig_acc = plt.figure(figsize=(5, 4))
    final_nn.plot_metrics(fig_loss=fig_loss, fig_acc=fig_acc)

    return final_nn, best_config

    # nn.plot_metrics(fig_loss=fig_loss, fig_acc=fig_acc, rows=n_rows, cols=n_cols, plot_index=i, changing_hyperpar=changing_hyperpar[i])

    # print(f"The best combination is:\n{trials[best_vl_loss[1]]}\n\nwith a vl loss of {best_vl_loss[0]}\n\n\n")


def launch_trial(conf, training_sets, input_units, verbose=True):
    if verbose:
        print(f"Parameters:\n{conf}\n")

    X_train, X_val, T_train, T_val = training_sets

    output_units = conf["architecture"]["output_units"]
    neurons_per_layer = conf["architecture"]["neurons_per_layer"]
    hidden_act_func = conf["functions"]["hidden"]
    hidden_act_param = conf["functions"]["hidden_param"]
    output_act_func = conf["functions"]["output"]
    output_act_param = conf["functions"]["output_param"]
    act_func = [[hidden_act_func, hidden_act_param], [output_act_func, output_act_param]]   

    train_args = conf["training"]
    training_hyperpar = conf["training"]

    early_stopping = conf["training"]["early_stopping"]

    loss_func = conf["functions"]["loss"]

    extractor = utils.create_extractor(conf["initialization"]["method"])

    nn = NeuralNetwork(num_inputs=input_units,
                       num_outputs=output_units,
                       neurons_per_layer=neurons_per_layer,
                       training_hyperpar=training_hyperpar,
                       extractor=extractor,
                       activation=act_func,
                       early_stopping=early_stopping)
    
    nn.train(X_train, T_train, X_val, T_val, train_args=train_args, loss_func=loss_func, early_stopping=early_stopping)

    best_vl_loss = nn.loss_calculator(X_val, T_val, losses.losses_functions[loss_func])
    if verbose:
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
 

def evaluate_configuration(trial_config, training_sets, input_units):
    """
    Evaluates a single hyperparameter configuration using either 
    Hold-Out or K-Fold CV based on the config.
    """
    vl_method = trial_config["validation"]["method"]

    if vl_method == "hold_out":
        loss, _ = launch_trial(trial_config, training_sets, input_units, verbose=False)
        return loss

    elif vl_method == "k_fold_cv":
        k_folds = trial_config["validation"]["folds"]
        
        # Combine sets for k fold
        X_full = np.concatenate((training_sets[0], training_sets[1]))
        T_full = np.concatenate((training_sets[2], training_sets[3]))
        
        indices = utils.get_k_fold_indices(len(X_full), k_folds)
        
        total_loss = 0

        # run over k folds  
        for i in range(k_folds):
            val_start, val_end = indices[i]
            
            X_val_k = X_full[val_start:val_end]
            T_val_k = T_full[val_start:val_end]
            
            X_train_k = np.concatenate([X_full[:val_start], X_full[val_end:]])
            T_train_k = np.concatenate([T_full[:val_start], T_full[val_end:]])
            
            current_sets = [X_train_k, X_val_k, T_train_k, T_val_k]
            
            loss, _ = launch_trial(trial_config, current_sets, input_units, verbose=False)
            total_loss += loss
            
        avg_loss = total_loss / k_folds
        print(f"Average {k_folds}-fold loss: {avg_loss:.6f}\n")
        return avg_loss

    else:
        raise ValueError("Unknown validation method")