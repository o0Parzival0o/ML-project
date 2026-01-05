import utils
from model import NeuralNetwork

import numpy as np
import matplotlib.pyplot as plt

import datetime
import os
import json
import itertools


def model_assessment(training_sets, input_units, config, test_sets=None):
    '''Depending on config, chooses betwen hold out or k fold cv assessment and estimates the risk on tests set
    '''
    X_training, T_training = training_sets
    method_assessment = config["assessment"]["method"]
    
    if method_assessment not in ["hold_out", "k_fold_cv", "leave_one_out_cv"]:
        raise ValueError(f"Unknown assessment method: {method_assessment}")
    
    if method_assessment == "hold_out":     #assessment with hold out
        if test_sets is None:
            data_split_prop = [config["training"]["splitting"]["tr"] + config["training"]["splitting"]["vl"], config["training"]["splitting"]["ts"]]
            X_train, X_test, T_train, T_test = utils.data_splitting(X_training, T_training, data_split_prop)

            train_set = [X_train, T_train]
            test_set = [X_test, T_test]
        else:
            train_set = [X_training, T_training]
            test_set = test_sets

        final_model, risk, accuracy = hold_out_assessment(config, input_units, train_set, test_set)     #run assessment, get risk and accuracy
        print(f"Test risk: {risk:.6f}")
        if accuracy is not None:
            print(f"Test accuracy: {accuracy:.2%}")
        return final_model, risk, accuracy
    
    else:       #assessment with k fold
        num_folds = config["assessment"]["folds"] if method_assessment == "k_fold_cv" else len(X_training)

        final_model, avg_risk, std_risk, avg_accuracy, std_accuracy = k_fold_assessment(num_folds, config, input_units, [X_training, T_training])       #run assessment, get risk and accuracy
        print(f"Average {num_folds}-fold test risk: {avg_risk:.6f} ± {std_risk:.6f}")
        if avg_accuracy is not None:
            print(f"Average {num_folds}-fold test accuracy: {avg_accuracy:.2%} ± {std_accuracy:.2%}\n")
        return final_model, avg_risk, avg_accuracy


def perform_search(X_train, T_train, input_units, config):
    '''High level orchestrator, calls grid or random search, returns best config, best nn, and loss. 
    Does not handle logic to iterate between configurations, that it is up to grid_search or random_search
    '''
    search_type = config["training"]["search_type"]

    if search_type not in ["grid", "random"]:
        raise ValueError('"search_type" must be "grid" or "random"')

    if search_type == "grid":
        best_config, best_nn, best_idx, avg_loss, fig_loss, fig_acc = grid_search(X_train, T_train, input_units, config)

    if search_type == "random":
        best_config, best_nn, best_idx, avg_loss, fig_loss, fig_acc = random_search(X_train, T_train, input_units, config)

    print(f"Miglior configurazione scelta: trial {best_idx + 1}")
    print(f"{best_config}\n")

    return best_config, best_nn, avg_loss, fig_loss, fig_acc


def grid_search(X_training, T_training, input_units, config):
    '''Runs grid search with parameters from a given config on a given dataset
    '''
    flattened_values = utils.flatten_config(config)
    keys, values = zip(*flattened_values.items())

    trials = []
    for comb in itertools.product(*values):         #creates the cartesian product with all the combinations of the hyperparameters read from json
        new_config = {}     
        for k, v in zip(keys, comb):
            utils.set_dict(new_config, k, v)            #select a combination of values and create a configuration in the same format as the json such that in can be used for training
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
    
    # identify which parameter changes to be able to plot it afterwards
    changing_hyperpar = []
    for trial in flattened_trials:
        trial_changing = {k: trial[k] for k in changing_keys}
        changing_hyperpar.append(trial_changing)

    num_trials = len(trials)
    n_cols = int(np.ceil(np.sqrt(num_trials)))
    n_rows = int(np.ceil(num_trials / n_cols))
    
    fig_loss = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    fig_acc = plt.figure(figsize=(5 * n_cols, 4 * n_rows))

    # run all the possible trials
    best_vl_loss = float("inf")
    best_trial_idx = None
    nn_list = []
    avg_loss = None
    for i, trial in enumerate(trials):          #run a specific trial
        print(f"Trial {i+1}/{len(trials)} :")

        nn, loss, accuracy = model_selection(trial, X_training, T_training, input_units)            #perform model selection on a specific trial configuration and get its loss
        nn_list.append(nn)

        if loss is None or np.isnan(loss):
            print("Loss is None or nan\n")
            continue
        
        if loss < best_vl_loss:             #if loss is better than previous best, update all the references to the best loss
            best_vl_loss = loss
            avg_loss = loss
            best_trial_idx = i
            print(f"Loss: {loss:.6f}")
            if accuracy is not None:
                print(f"(accuracy: {accuracy:.2%})")
            print(f"New best found!\n")
        else:
            print(f"Loss: {loss:.6f}")
            if accuracy is not None:
                print(f"(accuracy: {accuracy:.2%})\n")
            else:
                print("")

        nn.plot_metrics(fig_loss, fig_acc, n_rows, n_cols, plot_index=i, changing_hyperpar=changing_hyperpar[i], title="grid_search", data_type=config["general"]["dataset_name"])

    best_config = trials[best_trial_idx]
    best_nn = nn_list[best_trial_idx]

    return best_config, best_nn, best_trial_idx, avg_loss, fig_loss, fig_acc


def random_search(X_training, T_training, input_units, config):
    '''Runs random search with a given parameters config on a given dataset
    '''
    flattened_values = utils.flatten_config(config)
    keys, values = zip(*flattened_values.items())

    hyperpar_error = []
    hyperpar_bounded = []
    hyperpar_constant = []
    for idx, val in enumerate(values):          #get hyperparameter values bounds or return error
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

            if k in hyperpar_bounded:               #make a random extraction for each hyperparameter bounds
                if k == "training.regularization":
                    random_value = utils.loguniform(v[0], v[1])
                elif isinstance(v[0], int) and isinstance(v[1], int):
                    random_value = np.random.randint(v[0], v[1] + 1)
                elif isinstance(v[0], float) or isinstance(v[1], float):
                    random_value = np.random.uniform(v[0], v[1]) 
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

    num_trials = len(trials)
    n_cols = int(np.ceil(np.sqrt(num_trials)))
    n_rows = int(np.ceil(num_trials / n_cols))
    
    fig_loss = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    fig_acc = plt.figure(figsize=(5 * n_cols, 4 * n_rows))

    # run all the possible trial
    best_vl_loss = float("inf")
    best_trial_idx = None
    nn_list = []
    avg_loss = None
    for i, trial in enumerate(trials):          #run a specific trial
        print(f"Trial {i+1}/{len(trials)} :")

        nn, loss, accuracy = model_selection(trial, X_training, T_training, input_units)            #perform model selection on a specific trial configuration and get its loss
        nn_list.append(nn)

        if loss is None:
            print("Loss is None\n")
            continue

        if loss < best_vl_loss:         #if loss is better than previous best, update all the references to the best loss
            best_vl_loss = loss
            avg_loss = loss
            best_trial_idx = i
            print(f"Loss: {loss:.6f}")
            if accuracy is not None:
                print(f"(accuracy: {accuracy:.2%})")
            print(f"New best found!\n")
        else:
            print(f"Loss: {loss:.6f}")
            if accuracy is not None:
                print(f"(accuracy: {accuracy:.2%})\n")
            else:
                print("")
        
        nn.plot_metrics(fig_loss, fig_acc, n_rows, n_cols, plot_index=i, num_trials=num_trials, changing_hyperpar=changing_hyperpar[i], title="random_search", data_type=config["general"]["dataset_name"])

    best_config = trials[best_trial_idx]
    best_nn = nn_list[best_trial_idx]

    return best_config, best_nn, best_trial_idx, avg_loss, fig_loss, fig_acc


def launch_trial(conf, train_set, val_set, input_units, verbose=True):
    '''Runs a single training event (no model selection) with a given config'''
    if verbose:
        print(f"Parameters:\n{conf}\n")

    X_train, T_train = train_set
    X_val, T_val = val_set

    preprocess = conf["preprocessing"]["type"]          #perform preprocessing as specified in the config
    if preprocess == "standardization":
        X_mean, X_std = utils.standardization(X_train)
        T_mean, T_std = utils.standardization(T_train)
        X_train = (X_train - X_mean) / X_std
        T_train = (T_train - T_mean) / T_std
        X_val = (X_val - X_mean) / X_std
        T_val = (T_val - T_mean) / T_std

        X_params = (X_mean, X_std)
        T_params = (T_mean, T_std)

    elif preprocess == "rescaling":
        X_min, X_max = utils.scaling(X_train)
        T_min, T_max = utils.scaling(T_train)
        X_train = (X_train - X_min) / (X_max - X_min)
        T_train = (T_train - T_min) / (T_max - T_min)
        X_val = (X_val - X_min) / (X_max - X_min)
        T_val = (T_val - T_min) / (T_max - T_min)
    
        X_params = (X_min, X_max)
        T_params = (T_min, T_max)

    else:
        preprocess = None
        X_params = None
        T_params = None

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
    extractor = utils.create_extractor(conf["initialization"]["method"], conf["initialization"]["range"])

    preprocessing = [preprocess, X_params, T_params]

    nn = NeuralNetwork(num_inputs=input_units,
                       num_outputs=output_units,
                       neurons_per_layer=neurons_per_layer,
                       training_hyperpar=training_hyperpar,
                       extractor=extractor,
                       activation=act_func,
                       preprocessing=preprocessing,
                       )
    
    nn.train(X_train, T_train, X_val, T_val, train_args=train_args, loss_func=loss_func, early_stopping=early_stopping)         #launch training

    best_vl_loss = nn.best_loss         #best vl loss represents the lowest loss obtained during training on vl, keeping the lowest loss even if the training continues for $patience$ epochs without improving the loss. It represents a single run, and shall not be confused with the avg loss obtained in model selection by averaging losses of different folds
    best_vl_accuracy = nn.best_accuracy         #same as for the comment above
    if verbose:
        print(f"Best validation loss for this run: {best_vl_loss:.6f}\n")

    return nn, best_vl_loss, best_vl_accuracy


def model_selection(trial_config, X_train, T_train, input_units):
    '''Lower level selection, evaluates a configuration via hold out or k fold
    '''
    method_selection = trial_config["validation"]["method"]

    if method_selection not in ["hold_out", "k_fold_cv", "leave_one_out_cv"]:
        raise ValueError("Unknown validation method")

    if method_selection == "hold_out":      
        data_split_prop = [trial_config["training"]["splitting"]["tr"], trial_config["training"]["splitting"]["vl"]]
        X_train, X_val, T_train, T_val = utils.data_splitting(X_train, T_train, data_split_prop)            #splits data for hold out according to configuration file proportions

        train_set = [X_train, T_train]
        val_set = [X_val, T_val]

        nn, loss, accuracy = hold_out_selection(trial_config, input_units, train_set, val_set)          #run model selection with hold out vl
        return nn, loss, accuracy

    else:          
        num_folds = trial_config["validation"]["folds"] if method_selection == "k_fold_cv" else len(X_train)

        nn, avg_loss, std_loss, avg_accuracy, std_accuracy = k_fold_selection(num_folds, trial_config, input_units, [X_train, T_train])  #run model selection with k fold cv

        print(f"Average {num_folds}-fold loss: {avg_loss:.6f} ± {std_loss:.6f}")
        if avg_accuracy is not None:
            print(f"Average {num_folds}-fold accuracy: {avg_accuracy:.2%} ± {std_accuracy:.2%}\n")
        return nn, avg_loss, avg_accuracy


def train_final_model(config, X_train, T_train, X_test=None, T_test=None, input_units=None, epochs=None, target_loss=None):
    '''Retrains the model with the best parameters on tr+vl
    '''
    print("Retrain:")

    output_units = config["architecture"]["output_units"]
    neurons_per_layer = config["architecture"]["neurons_per_layer"]
    hidden_act_func = config["functions"]["hidden"]
    hidden_act_param = config["functions"]["hidden_param"]
    output_act_func = config["functions"]["output"]
    output_act_param = config["functions"]["output_param"]
    act_func = [[hidden_act_func, hidden_act_param], [output_act_func, output_act_param]]   
    train_args = config["training"]
    training_hyperpar = config["training"]
    loss_func = config["functions"]["loss"]
    extractor = utils.create_extractor(config["initialization"]["method"], config["initialization"]["range"])

    preprocess = config["preprocessing"]["type"]
    if preprocess == "standardization":
        X_mean, X_std = utils.standardization(X_train)
        T_mean, T_std = utils.standardization(T_train)
        X_train = (X_train - X_mean) / X_std
        T_train = (T_train - T_mean) / T_std
        if X_test is not None:
            X_test = (X_test - X_mean) / X_std
            T_test = (T_test - T_mean) / T_std

        X_params = (X_mean, X_std)
        T_params = (T_mean, T_std)

    elif preprocess == "rescaling":
        X_min, X_max = utils.scaling(X_train)
        T_min, T_max = utils.scaling(T_train)
        X_train = (X_train - X_min) / (X_max - X_min)
        T_train = (T_train - T_min) / (T_max - T_min)
        if X_test is not None:
            X_test = (X_test - X_min) / (X_max - X_min)
            T_test = (T_test - T_min) / (T_max - T_min)
    
        X_params = (X_min, X_max)
        T_params = (T_min, T_max)

    else:
        preprocess = None
        X_params = None
        T_params = None

    preprocessing = [preprocess, X_params, T_params]

    nn = NeuralNetwork(num_inputs=input_units,
                       num_outputs=output_units,
                       neurons_per_layer=neurons_per_layer,
                       training_hyperpar=training_hyperpar,
                       extractor=extractor,
                       activation=act_func,
                       preprocessing=preprocessing)
    
    if target_loss is not None:         #during final retraining, set stopping method to target loss. training will continue until the loss obtained on the vl set is reached
        config["training"]["early_stopping"] = {
            "enabled": True,
            "monitor": "train_loss",
            "patience": config["training"]["epochs"],
            "target_loss": target_loss
        }
        early_stopping = config["training"]["early_stopping"]
    elif epochs is not None:
        config["training"]["epochs"] = epochs
        early_stopping = {"enabled": False}
    else:
        early_stopping = {"enabled": False}
    
    nn.train(X_train, T_train, X_vl=None, T_vl=None, X_ts=X_test, T_ts=T_test, train_args=train_args,           #launch final training on tr+vl set
             loss_func=loss_func, early_stopping=early_stopping)
    
    print(f"Loss: {nn.best_loss:.6f}")
    if nn.best_accuracy is not None:
        print(f"(accuracy: {nn.best_accuracy:.2%})\n")

    return nn


def evaluate_model(nn: NeuralNetwork, test_set):
    '''Gets performance metrics score of a NN with a test set
    '''
    X_test, T_test = test_set
    predictions = np.array([nn.feed_forward(x) for x in X_test])
    risk = nn.loss_calculator(X_test, T_test)
    accuracy_risk = nn.accuracy_calculator(X_test, T_test)
    return risk, accuracy_risk, predictions


def hold_out_selection(config, input_units, train_set, val_set):

    nn, loss, accuracy = launch_trial(config, train_set, val_set, input_units, verbose=False)
    return nn, loss, accuracy


def hold_out_assessment(config, input_units, train_set, test_set):
    '''Runs search for best model, trains best model, evaluates it
    '''
    X_training, T_training = train_set
    X_test, T_test = test_set

    # model selection
    best_config, best_nn, avg_loss, fig_loss_search, fig_acc_search = perform_search(X_training, T_training, input_units, config)

    # retraining
    method_selection = config["validation"]["method"]
    if method_selection == "hold_out":
        final_model = train_final_model(best_config, X_training, T_training, X_test, T_test, input_units, epochs=best_nn.best_epoch)
    else:
        final_model = train_final_model(best_config, X_training, T_training, X_test, T_test, input_units, target_loss=avg_loss)

    # model assessment
    risk, accuracy_risk, _ = evaluate_model(final_model, test_set)

    save_choice = input("Do you want to save the model? (0: No; 1: Yes)\n")
    dir_name = input("Directory name: ")
    if save_choice == "1":
        dataset_name = config["general"]["dataset_name"]
        path = f"../../model_saved/{dataset_name}/" + (f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{dir_name}" if dir_name != "" else f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(path, exist_ok=True)

        model_path = os.path.join(path, "model.pkl")
        final_model.save_model(model_path)
        best_config_path = os.path.join(path, "best_config.json")
        with open(best_config_path, "w") as file:
            json.dump(best_config, file)
        config_path = os.path.join(path, "config.json")
        with open(config_path, "w") as file:
            json.dump(config, file)
        result_path = os.path.join(path, "result.txt")
        with open(result_path, "w") as file:
            file.write(f"Date:\t\t\t{datetime.datetime.now().isoformat()}\n")
            file.write(f"Retrain loss:\t{final_model.best_loss:.6f}\n")
            if final_model.best_accuracy is not None:
                file.write(f"Retrain accuracy:\t{final_model.best_accuracy:.2%}\n")
            file.write(f"Test Loss:\t\t{risk:.6f}\n")
            if accuracy_risk is not None:
                file.write(f"Test Accuracy:\t\t{accuracy_risk:.2%}\n")
            file.write(f"Best Epoch:\t\t{best_nn.best_epoch}")

        fig_loss = plt.figure(figsize=(5, 4))
        fig_acc = plt.figure(figsize=(5, 4))
        final_model.plot_metrics(fig_loss=fig_loss, fig_acc=fig_acc, title="best_model", data_type=dataset_name, save_path=path)
        
        search_type = config["training"]["search_type"]
        if fig_loss_search is not None:
            fig_loss_search.savefig(f'{path}/{dataset_name}_{search_type}_search_loss.png', dpi=300)
            plt.close(fig_loss_search)
        if fig_acc_search is not None and final_model.output_activation.__name__ in ["sigmoid", "tanh"]:
            fig_acc_search.savefig(f'{path}/{dataset_name}_{search_type}_search_accuracy.png', dpi=300)
            plt.close(fig_acc_search)

    return final_model, risk, accuracy_risk


def k_fold_selection(k, config, input_units, train_set):
    '''Creates the folds and calculates best model on avg loss on a given config'''    
    k_folds = k
    indices = utils.get_k_fold_indices(len(train_set[0]), k_folds)

    total_loss = []
    total_accuracy = []
    nn_list = []
    # run over k folds  
    for i in range(k_folds):
        val_start, val_end = indices[i]
        
        X_val_k = np.array(train_set[0][val_start:val_end])
        T_val_k = np.array(train_set[1][val_start:val_end])
        
        X_train_k = np.concatenate([train_set[0][:val_start], train_set[0][val_end:]])
        T_train_k = np.concatenate([train_set[1][:val_start], train_set[1][val_end:]])
        
        current_train_set = [X_train_k, T_train_k]
        current_val_test_set = [X_val_k, T_val_k]

        nn, loss, accuracy = launch_trial(config, current_train_set, current_val_test_set, input_units, verbose=False)

        total_loss.append(loss)
        if accuracy is not None:
            total_accuracy.append(accuracy)
        nn_list.append(nn)
        
    avg_loss = np.mean(total_loss)
    std_loss = np.std(total_loss)
    if len(total_accuracy) != 0:
        avg_accuracy = np.mean(total_accuracy)
        std_accuracy = np.std(total_accuracy)
    else:
        avg_accuracy = None
        std_accuracy = None

    # save best model of kfold only for plotting (debug)
    best_fold_idx = np.argmin(total_loss)
    best_nn = nn_list[best_fold_idx]

    return best_nn, avg_loss, std_loss, avg_accuracy, std_accuracy


def k_fold_assessment(k, config, input_units, train_set):
    '''Creates folds and launches the search for the best model so to do double nested cv, then evaluates it
    '''
    k_folds = k
    indices = utils.get_k_fold_indices(len(train_set[0]), k_folds)

    total_risk = []
    total_accuracy = []
    nn_list = []
    all_fig_loss_search = []
    all_fig_acc_search = []
    # run over k folds  
    for i in range(k_folds):
        print(f"External fold {i+1}/{k_folds} :")
        test_start, test_end = indices[i]
        
        X_test_k = train_set[0][test_start:test_end]
        T_test_k = train_set[1][test_start:test_end]
        
        X_train_k = np.concatenate([train_set[0][:test_start], train_set[0][test_end:]])
        T_train_k = np.concatenate([train_set[1][:test_start], train_set[1][test_end:]])
        
        # model selection
        best_config, best_nn, avg_loss, fig_loss_search, fig_acc_search = perform_search(X_train_k, T_train_k, input_units, config)
        
        all_fig_loss_search.append(fig_loss_search)
        all_fig_acc_search.append(fig_acc_search)

        # retraining
        method_selection = config["validation"]["method"]
        if method_selection == "hold_out":
            final_model = train_final_model(best_config, X_train_k, T_train_k, X_test_k, T_test_k, input_units, epochs=best_nn.best_epoch)
        else:
            final_model = train_final_model(best_config, X_train_k, T_train_k, X_test_k, T_test_k, input_units, target_loss=avg_loss)

        # model assessment
        risk, accuracy, _ = evaluate_model(final_model, [X_test_k, T_test_k])

        print(f"Risk: {risk:.6f}")
        if accuracy is not None:
            print(f"(accuracy: {accuracy:.2%})\n")

        total_risk.append(risk)
        if accuracy is not None:
            total_accuracy.append(accuracy)
        nn_list.append(final_model)
        
    avg_risk = np.mean(total_risk)
    std_risk = np.std(total_risk)
    if len(total_accuracy) != 0:
        avg_accuracy = np.mean(total_accuracy)
        std_accuracy = np.std(total_accuracy)
    else:
        avg_accuracy = None
        std_accuracy = None

    # save best model of kfold only for plotting (debug)
    best_fold_idx = np.argmin(total_risk)
    best_nn = nn_list[best_fold_idx]

    save_choice = input("Do you want to save the model? (0: No; 1: Yes)\n")
    dir_name = input("Directory name: ")
    if save_choice == "1":
        dataset_name = config["general"]["dataset_name"]
        path = f"../../model_saved/{dataset_name}/" + (f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{dir_name}" if dir_name != "" else f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(path, exist_ok=True)

        model_path = os.path.join(path, "model.pkl")
        final_model.save_model(model_path)
        best_config_path = os.path.join(path, "best_config.json")
        with open(best_config_path, "w") as file:
            json.dump(best_config, file)
        config_path = os.path.join(path, "config.json")
        with open(config_path, "w") as file:
            json.dump(config, file)
        result_path = os.path.join(path, "result.txt")
        with open(result_path, "w") as file:
            file.write(f"Date:\t\t\t{datetime.datetime.now().isoformat()}\n")
            file.write(f"Retrain loss:\t{final_model.best_loss:.6f}\n")
            if final_model.best_accuracy is not None:
                file.write(f"Retrain accuracy:\t{final_model.best_accuracy:.2%}\n")
            file.write(f"Test Loss:\t\t{avg_risk:.6f} ± {std_risk:.6f}\n")
            if avg_accuracy is not None:
                file.write(f"Test Accuracy:\t\t{avg_accuracy:.2%} ± {std_accuracy:.2%}\n")
            file.write(f"Best Epoch:\t\t{best_nn.best_epoch}")
        
        fig_loss = plt.figure(figsize=(5, 4))
        fig_acc = plt.figure(figsize=(5, 4))
        best_nn.plot_metrics(fig_loss=fig_loss, fig_acc=fig_acc, title="best_model", data_type=dataset_name, save_path=path)
        
        search_type = config["training"]["search_type"]
        for fold_idx, (fig_loss_s, fig_acc_s) in enumerate(zip(all_fig_loss_search, all_fig_acc_search)):
            if fig_loss_s is not None:
                fig_loss_s.savefig(f'{path}/{dataset_name}_{search_type}_search_fold{fold_idx+1}_loss.png', dpi=300)
                plt.close(fig_loss_s)
            if fig_acc_s is not None and best_nn.output_activation.__name__ in ["sigmoid", "tanh"]:
                fig_acc_s.savefig(f'{path}/{dataset_name}_{search_type}_search_fold{fold_idx+1}_accuracy.png', dpi=300)
                plt.close(fig_acc_s)


    return best_nn, avg_risk, std_risk, avg_accuracy, std_accuracy
