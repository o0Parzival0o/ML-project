import utils
from model import NeuralNetwork
from data_loader import data_loader
from model_selection import perform_search

import matplotlib.pyplot as plt

import random
import time



if __name__ == "__main__":

    
    start = time.time()

    config = utils.load_config_json("config.json")
    random.seed(config["general"]["seed"])

    train_args = config["training"]

    # MONK DATASET
    monk_train_data = config["paths"]["MONK_train_data"]
    monk_test_data = config["paths"]["MONK_test_data"]
    X_train, T_train, input_units = data_loader(monk_train_data, shuffle=True)       
    X_test, T_test, _ = data_loader(monk_test_data, shuffle=False)


    # # EXAM DATASET
    # CUP_train_data = config["paths"]["CUP_train_data"]
    # X_train, t_train, input_units = data_loader(CUP_train_data, shuffle=False)
    # X_CUP, t_CUP, input_units = data_loader(CUP_test_data, shuffle=False)

    hidden_act_func = config["functions"]["hidden"]
    output_act_func = config["functions"]["output"]
    act_func = [hidden_act_func, output_act_func]                   #TODO mandare al modello nei training hyperpar o in un altro modo
    
    training_hyperpar = config["training"]
    early_stopping = config["training"]["early_stopping"]

    loss_func = config["functions"]["loss"]

    extractor = utils.create_random_extractor(config["initialization"]["method"])

    # data_split_prop = [training_hyperpar["splitting"]["tr"], training_hyperpar["splitting"]["vl"], training_hyperpar["splitting"]["ts"]]
    # X_train, X_val, _, T_train, T_val, _ = utils.data_splitting(X_train, T_train, data_split_prop)

    data_split_prop = [0.7, 0.3]
    X_train, X_val, T_train, T_val = utils.data_splitting(X_train, T_train, data_split_prop)

    training_sets = [X_train, X_val, T_train, T_val]
    # nn = NeuralNetwork(num_inputs=input_units,
    #                    num_outputs=config["architecture"]["output_units"],
    #                    neurons_per_layer=config["architecture"]["neurons_per_layer"],
    #                    training_hyperpar=training_hyperpar,
    #                    extractor=extractor,
    #                    activation=act_func,
    #                    early_stopping=early_stopping)
    
    # nn.train(X_train, T_train, X_val, T_val, train_args=train_args, loss_func=loss_func, early_stopping=early_stopping)
    # nn.test(X_test, T_test)
    

    #TODO nn validate da fare
    #TODO uniformare impostazione seed randomico da json nei vari file py, non so se sia per come funziona rand ma per ora l'inizializzazione mi sembra essere diversa tra run diverse
    
    search_type = "grid"
    perform_search(training_sets, input_units, search_type)


    

    end = time.time() - start
    print(f"Elapsed time: {end} s")

    # fig1 = plt.figure(figsize=(5, 4))
    # fig2 = plt.figure(figsize=(5, 4))
    # nn.plot_metrics(fig1, fig2)
    

    #print(nn)
    #nn.plot()                                   # (da eliminare prima di mandare a Micheli)

    #TODO forse ha senso rimuovere dai config il seme (tanto basta far sì che sia riproducibile con seme hardcodato su macchine diverse, non ci interessa cambiare il seme, oppure vedere se ha senso tenerlo e provare con inizializzazioni diverse)