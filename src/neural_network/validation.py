import utils
import activations as actfun
import losses
import itertools
from model import NeuralNetwork

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def grid_search(training_sets,input_units):
    config = utils.load_config_json("vl_config.json")
    
    # output_units = config["architecture"]
    # neurons_per_layer


    flattened_values = utils.flatten_config(config)

    
    keys, values = zip(*flattened_values.items())

    # print(keys,values)

    trials = []

    for comb in itertools.product(*values):
        new_config = {}

        vl_loss = [float("inf"),-1]
        for k,v in zip(keys,comb):
            utils.set_dict(new_config,k,v)
        trials.append(new_config)

        
    for i,trial in zip((range(len(trials))),trials):


        loss = launch_trial(trial,training_sets,input_units)
        if(loss < vl_loss[0]):
            vl_loss = loss,i

    print(f"The best combination is {trials[vl_loss[1]]}\nwith a vl loss of {vl_loss[0]}")


def launch_trial(comb,training_sets,input_units):
    print(f"Lancio run con parametri {comb}\n\n\n")

    X_train,X_val,T_train,T_val = training_sets

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
                       num_outputs=comb["architecture"]["output_units"],
                       neurons_per_layer=comb["architecture"]["neurons_per_layer"],
                       training_hyperpar=training_hyperpar,
                       extractor=extractor,
                       activation=act_func,
                       early_stopping=early_stopping)
    
    nn.train(X_train, T_train,X_val,T_val, train_args=train_args, loss_func=loss_func,early_stopping=early_stopping)

    vl_loss = nn.validation_loss(X_val,T_val,loss_func)
    print(f"\n loss for this run {vl_loss}\n")

    return vl_loss

    # nn.test(X_val, T_val)


def perform_search(training_sets,input_units,search_type):
    if(search_type == "grid"):
        grid_search(training_sets,input_units)
