import utils
from model import NeuralNetwork
from data_loader import data_loader
from model_selection import model_assessment

import matplotlib.pyplot as plt
import numpy as np

import time



if __name__ == "__main__":

    data = "MONK"
    single_trial = False

    start = time.time()


    if data == "MONK":
        config = utils.load_config_json("../../config_files/MONK_config.json" if single_trial else "../../config_files/MONK_model_selection_config.json")
        np.random.seed(config["general"]["seed"])

        monk_train_data = config["paths"]["train_data"]
        monk_test_data = config["paths"]["test_data"]
        X_train, T_train, input_units = data_loader(monk_train_data, data_type="MONK", shuffle=True)       
        X_test, T_test, _ = data_loader(monk_test_data, data_type="MONK", shuffle=True)

        if single_trial:

            data_split_prop = [config["training"]["splitting"]["tr"], config["training"]["splitting"]["vl"]]
            X_train, X_val, T_train, T_val = utils.data_splitting(X_train, T_train, data_split_prop)

            train_args = config["training"]

            hidden_act_func = config["functions"]["hidden"]
            hidden_act_param = config["functions"]["hidden_param"]
            output_act_func = config["functions"]["output"]
            output_act_param = config["functions"]["output_param"]
            act_func = [[hidden_act_func, hidden_act_param], [output_act_func, output_act_param]]
            
            training_hyperpar = config["training"]
            early_stopping = config["training"]["early_stopping"]

            loss_func = config["functions"]["loss"]

            extractor = utils.create_extractor(config["initialization"]["method"])

            nn = NeuralNetwork(num_inputs=input_units,
                            num_outputs=config["architecture"]["output_units"],
                            neurons_per_layer=config["architecture"]["neurons_per_layer"],
                            training_hyperpar=training_hyperpar,
                            extractor=extractor,
                            activation=act_func)
            
            nn.train(X_train, T_train, X_val, T_val, train_args=train_args, loss_func=loss_func, early_stopping=early_stopping)
            nn.test(X_test, T_test)

            fig1 = plt.figure(figsize=(5, 4))
            fig2 = plt.figure(figsize=(5, 4))
            nn.plot_metrics(fig1, fig2, title="single_try")

        else:
            training_sets = [X_train, T_train]
            test_sets = [X_test, T_test]
            nn, _, _ = model_assessment(training_sets,input_units, config, test_sets)
            nn.save_model("../../model_saved/MONK_model.pkl")


    else:
        config = utils.load_config_json("../../config_files/config.json" if single_trial else "../../config_files/model_selection_config.json")
        np.random.seed(config["general"]["seed"])

        CUP_train_data = config["paths"]["train_data"]
        CUP_test_data = config["paths"]["test_data"]
        X_train, T_train, input_units = data_loader(CUP_train_data, data_type="train", shuffle=True)
        X_CUP, _, input_units_CUP = data_loader(CUP_test_data, data_type="test", shuffle=False)

        # utils.plot_dataset(X_train, T_train, X_CUP)

        # X_mean, X_std = utils.standardization(X_train)
        # T_mean, T_std = utils.standardization(T_train)
        # X_train = (X_train - X_mean) / X_std
        # T_train = (T_train - T_mean) / T_std
        # # X_CUP = (X_CUP - X_mean) / X_std              # remember to do the inverse at the end with "inverse_scaling"

        X_min, X_max = utils.scaling(X_train)
        T_min, T_max = utils.scaling(T_train)
        X_train = (X_train - X_min) / (X_max - X_min)
        T_train = (T_train - T_min) / (T_max - T_min)
        # X_CUP = (X_CUP - X_min) / (X_max - X_min)

        # utils.plot_correlation(X_train, T_train)

        if single_trial:

            data_split_prop = [config["training"]["splitting"]["tr"], config["training"]["splitting"]["vl"], config["training"]["splitting"]["ts"]]
            X_train, X_val, X_test, T_train, T_val, T_test = utils.data_splitting(X_train, T_train, data_split_prop)

            train_args = config["training"]

            hidden_act_func = config["functions"]["hidden"]
            hidden_act_param = config["functions"]["hidden_param"]
            output_act_func = config["functions"]["output"]
            output_act_param = config["functions"]["output_param"]
            act_func = [[hidden_act_func, hidden_act_param], [output_act_func, output_act_param]]
            
            training_hyperpar = config["training"]
            early_stopping = config["training"]["early_stopping"]

            loss_func = config["functions"]["loss"]

            extractor = utils.create_extractor(config["initialization"]["method"])

            nn = NeuralNetwork(num_inputs=input_units,
                               num_outputs=config["architecture"]["output_units"],
                               neurons_per_layer=config["architecture"]["neurons_per_layer"],
                               training_hyperpar=training_hyperpar,
                               extractor=extractor,
                               activation=act_func)
            
            nn.train(X_train, T_train, X_val, T_val, train_args=train_args, loss_func=loss_func, early_stopping=early_stopping)
            # nn.test(X_test, T_test)

            fig1 = plt.figure(figsize=(5, 4))
            fig2 = plt.figure(figsize=(5, 4))
            nn.plot_metrics(fig1, fig2, title="single_try")
        
        else:
            training_sets = [X_train, T_train]
            nn, _, _ = model_assessment(training_sets, input_units, config)
            nn.save_model("../../model_saved/MONK_model.pkl")


    end = time.time() - start
    print(f"Elapsed time: {end} s")


    # T_CUP = utils.inverse_standardization(T_CUP)

    #print(nn)
    #nn.plot()                                   # (da eliminare prima di mandare a Micheli)

    #TODO forse ha senso rimuovere dai config il seme (tanto basta far sì che sia riproducibile con seme hardcodato su macchine diverse, non ci interessa cambiare il seme, oppure vedere se ha senso tenerlo e provare con inizializzazioni diverse)
    #TODO nn validate da fare
    #TODO uniformare impostazione seed randomico da json nei vari file py, non so se sia per come funziona rand ma per ora l'inizializzazione mi sembra essere diversa tra run diverse
    
