import utils
import activations as actfun
import losses
import itertools
from model import NeuralNetwork

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def grid_search(training_sets, input_units):
    config = utils.load_config_json("model_selection_config.json")
    
    # output_units = config["architecture"]
    # neurons_per_layer

    flattened_values = utils.flatten_config(config)

    keys, values = zip(*flattened_values.items())

    # print(keys,values)

    trials = []
    for comb in itertools.product(*values):
        new_config = {}

        for k, v in zip(keys, comb):
            utils.set_dict(new_config, k, v)
        trials.append(new_config)

    num_trials = len(trials)
    n_cols = int(np.ceil(np.sqrt(num_trials)))
    n_rows = int(np.ceil(num_trials / n_cols))
    
    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    
    best_vl_loss = []
    networks_tried = []
    
    for i, trial in enumerate(trials):
        loss, nn = launch_trial(trial, training_sets, input_units, i)
        networks_tried.append((loss, nn))
        
        best_vl_loss = [loss, i] if not best_vl_loss or loss < best_vl_loss[0] else best_vl_loss


    # plotting the grid search of losses        
    for i, (loss, nn) in enumerate(networks_tried):        
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        
        ax.plot(nn.tr_loss, color='r', linestyle='-', label='Training')
        if nn.vl_loss:
            ax.plot(nn.vl_loss, color='b', linestyle='--', label='Validation')

        trial_config = trials[i]
        params_str = f"neurons: {trial_config['architecture']['neurons_per_layer']} / "
        params_str += f"lr: {trial_config['training']['learning_rate']}"
        
        ax.set_xlabel("Epochs", fontsize=8)
        ax.set_ylabel("Loss", fontsize=8)
        ax.set_title(f"Trial {i+1} (VL: {loss:.4f}): {params_str}", fontsize=8, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.show()
    
    print(best_vl_loss)

    print(f"The best combination is:\n{trials[best_vl_loss[1]]}\n\nwith a vl loss of {best_vl_loss[0]}\n\n\n")
    #TODO ricordarsi di rifare training del modello con parametri ottimi dopo la vl


def launch_trial(comb, training_sets, input_units, trial_num):
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

    vl_loss = nn.loss_calculator(X_val, T_val, losses.losses_functions[loss_func])
    print(f"Validation loss for this run: {vl_loss:.6f}\n")

    return vl_loss, nn


def perform_search(training_sets, input_units, search_type):
    if search_type == "grid":
        return grid_search(training_sets, input_units)
    else:
        raise ValueError('"search_type" must be "grid" or "random"')