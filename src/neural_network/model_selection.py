import utils
import losses
import itertools
from model import NeuralNetwork

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


def model_assessment(training_set, input_units, config):
    X_training = training_set[0]
    T_training = training_set[1]
    method_assessment = config["assessment"]["method"]
    
    if method_assessment not in ["hold_out", "k_fold_cv", "leave_one_out_cv"]:
        raise ValueError(f"Unknown assessment method: {method_assessment}")
    
    if method_assessment == "hold_out":
        data_split_prop = [config["training"]["splitting"]["tr"] + config["training"]["splitting"]["vl"], config["training"]["splitting"]["ts"]]
        X_train, X_test, T_train, T_test = utils.data_splitting(X_training, T_training, data_split_prop)

        train_set = [X_train, T_train]
        test_set = [X_test, T_test]

        risk, _ = hold_out(config, input_units, train_set, test_set, mod_type="assessment")
        print(f"Test risk: {risk:.6f}")
        return risk
    
    else:
        num_folds = config["validation"]["folds"] if method_assessment == "k_fold_cv" else len(X_train)

        avg_risk, std_risk = k_fold(num_folds, config, input_units, [X_training, T_training], mod_type="assessment")
        print(f"Average {num_folds}-fold test risk: {avg_risk:.6f} +- {std_risk:.6f}\n")
        return avg_risk
    

def perform_search(X_training, T_training, input_units, config):
    search_type = config["training"]["search_type"]
    if search_type == "grid":
        nn = grid_search(X_training, T_training, input_units, config)
        return nn
    if search_type == "random":
        nn = random_search(X_training, T_training, input_units, config)
        return nn
    else:
        raise ValueError('"search_type" must be "grid" or "random"')


def grid_search(X_training, T_training, input_units, config):

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

    # run all the possible trial
    best_vl_loss = None
    best_trial_idx = None
    losses = []
    # networks_tried = []
    for i, trial in enumerate(trials):
        print(f"Trial {i+1}/{len(trials)} :\n")

        loss = model_selection(trial, X_training, T_training, input_units)

        losses.append(loss)

        if best_vl_loss is None or loss < best_vl_loss:
            best_vl_loss = loss
            best_trial_idx = i
            print(f"New best loss: {best_vl_loss:.6f}\n")

    best_config = trials[best_trial_idx]

    # retrain on model with the best loss
    _, final_nn = launch_trial(best_config, [X_training, T_training], [X_training, T_training], input_units, verbose=True)

    fig_loss = plt.figure(figsize=(5, 4))
    fig_acc = plt.figure(figsize=(5, 4))
    final_nn.plot_metrics(fig_loss=fig_loss, fig_acc=fig_acc)

    return final_nn, best_config

def random_search(X_training, T_training, input_units, config):

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

    # run all the possible trial
    best_vl_loss = None
    best_trial_idx = None
    # networks_tried = []
    for i, trial in enumerate(trials):
        print(f"Trial {i+1}/{len(trials)} :\n")

        loss = model_selection(trial, X_training, T_training, input_units)

        if best_vl_loss is None or loss < best_vl_loss:
            best_vl_loss = loss
            best_trial_idx = i
            print(f"New best found! Loss: {best_vl_loss:.6f}")

    best_config = trials[best_trial_idx]

    # retrain on model with the best loss
    _, final_nn = launch_trial(best_config, [X_training, T_training], [X_training, T_training], input_units, verbose=True)

    fig_loss = plt.figure(figsize=(5, 4))
    fig_acc = plt.figure(figsize=(5, 4))
    final_nn.plot_metrics(fig_loss=fig_loss, fig_acc=fig_acc)

    return final_nn, best_config


def launch_trial(conf, train_set, val_set, input_units, verbose=True):
    if verbose:
        print(f"Parameters:\n{conf}\n")

    X_train, T_train = train_set
    X_val, T_val = val_set

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

    best_vl_loss = None
    if X_val is not None:
        best_vl_loss = nn.loss_calculator(X_val, T_val, losses.losses_functions[loss_func])
        if verbose:
            print(f"Best validation loss for this run: {best_vl_loss:.6f}\n")

    return best_vl_loss, nn



def model_selection(trial_config, X_train, T_train, input_units):
    """
    Evaluates a single model (hyperparameter configuration) using either Hold-Out or K-Fold CV based on the config.
    """
    method_selection = trial_config["validation"]["method"]

    if method_selection not in ["hold_out", "k_fold_cv", "leave_one_out_cv"]:
        raise ValueError("Unknown validation method")

    if method_selection == "hold_out":
        data_split_prop = [trial_config["training"]["splitting"]["tr"], trial_config["training"]["splitting"]["vl"]]
        X_train, X_val, T_train, T_val = utils.data_splitting(X_train, T_train, data_split_prop)

        train_set = [X_train, T_train]
        val_set = [X_val, T_val]

        loss = hold_out(trial_config, input_units, train_set, val_set)
        return loss

    else:
        num_folds = trial_config["validation"]["folds"] if method_selection == "k_fold_cv" else len(X_train)

        avg_loss, std_loss = k_fold(num_folds, trial_config, input_units, [X_train, T_train])
        print(f"Average {num_folds}-fold loss: {avg_loss:.6f} +- {std_loss:.6f}\n")
        return avg_loss
    
def evaluate_model(nn : NeuralNetwork, test_set, loss_func):
    X_test = test_set[0]
    T_test = test_set[1]
    predictions = np.array([nn.feed_forward(x) for x in X_test])
    risk = nn.loss_calculator(X_test, T_test, loss_func)
    return risk, predictions

def hold_out(config, input_units, train_set, val_test_set, mod_type=None):

    if mod_type == "assessment":
        loss_func = losses.losses_functions[config["functions"]["loss"]]
        nn, _ = perform_search(train_set[0], train_set[1], input_units, config)
        risk, _ = evaluate_model(nn, val_test_set, loss_func)
        return risk

    else:
        loss, _ = launch_trial(config, train_set, val_test_set, input_units, verbose=False)
        return loss

def k_fold(k, config, input_units, train_set, mod_type=None):

    k_folds = k
    indices = utils.get_k_fold_indices(len(train_set[0]), k_folds)

    total_loss_risk = []
    # run over k folds  
    for i in range(k_folds):
        val_test_start, val_test_end = indices[i]
        
        X_val_test_k = train_set[0][val_test_start:val_test_end]
        T_val_test_k = train_set[1][val_test_start:val_test_end]
        
        X_train_k = np.concatenate([train_set[0][:val_test_start], train_set[0][val_test_end:]])
        T_train_k = np.concatenate([train_set[1][:val_test_start], train_set[1][val_test_end:]])
        
        current_train_set = [X_train_k, T_train_k]
        current_val_test_set = [X_val_test_k, T_val_test_k]

        if mod_type == "assessment":
            loss_func = losses.losses_functions[config["functions"]["loss"]]
            nn, _ = perform_search(current_train_set[0], current_train_set[1], input_units, config)
            loss_risk, _ = evaluate_model(nn, current_val_test_set, loss_func)

        else:
            loss_risk = launch_trial(config, current_train_set, current_val_test_set, input_units, verbose=False)

        total_loss_risk.append(loss_risk)
        
    avg_loss = np.mean(total_loss_risk)
    std_loss = np.std(total_loss_risk)
    return avg_loss, std_loss