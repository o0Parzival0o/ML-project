import utils
import activations as actfun
import losses
import itertools

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def grid_search():
    config = utils.load_config_json("vl_config.json")
    
    # output_units = config["architecture"]
    # neurons_per_layer


    flattened_values = utils.flatten_config(config)
    
    keys, values = zip(*flattened_values.items())

    print(keys,values)

    trials = []

    for comb in itertools.product(*values):
        # print(comb)

        new_config = {}
        score = 0
        for k,v in zip(keys,comb):
            utils.set_dict(new_config,k,v)
        trials.append(new_config)
        # score = launch_trial(comb)
    for i,trial in zip(enumerate(range(len(trials))),trials):
        print(f"run numero {i} con dati {trial}\n\n")

# def launch_trial(comb):

