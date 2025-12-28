import utils
from model import NeuralNetwork
from data_loader import data_loader
from model_selection import model_assessment

import matplotlib.pyplot as plt
import numpy as np

import time



if __name__ == "__main__":

    data_input = input("Select which data woul you use. (1: MONK; 2: CUP)\n")
    if data_input == "1":
        data = "MONK"
    elif data_input == "2":
        data = "CUP"
    else:
        raise ValueError("Your input must be a number and between 1 and 2")

    if data == "MONK":
        selected = input("Select type of MONK (1, 2, 3).\n")
        if selected is not int and selected not in ["1","2","3"]:
            raise ValueError("Your input must be a number and between 1 and 3")
    elif data == "CUP":
        preprocess = input("Do you want to normalize the dataset? (0: No; 1: Standardize; 2: Rescaling)\n")
        if preprocess is not int and preprocess not in ["0","1","2"]:
            raise ValueError("Your input must be a number and between 0 and 2")

    prediction_input = input("Do you want to do a prediction? (0: No; 1: Yes)\n")
    if prediction_input == "0":
        prediction = False
        trial_input = input("Do you want to do a single trial or model selection (and model assessment)? (1: Single trial; 2: Model selection)\n")
        if trial_input == "1":
            single_trial = True
        elif trial_input == "2":
            single_trial = False
        else:
            raise ValueError("Your input must be a number and between 0 and 1")
    elif prediction_input == "1":
        prediction = True
    else:
        raise ValueError("Your input must be a number and between 0 and 1")

    start = time.time()

    if not prediction:
        if data == "MONK":
            config = utils.load_config_json(f"../../config_files/MONK_{selected}_config.json" if single_trial else f"../../config_files/MONK_{selected}_model_selection_config.json")
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

                fig1 = plt.figure(figsize=(5, 4))
                fig2 = plt.figure(figsize=(5, 4))
                nn.plot_metrics(fig1, fig2, title="single_try", data_type=f"MONK_{selected}")

            else:
                training_sets = [X_train, T_train]
                test_sets = [X_test, T_test]
                nn, _, _ = model_assessment(training_sets,input_units, config, test_sets)
                nn.save_model(f"../../model_saved/MONK_{selected}_model.pkl")
                nn.plot(f"MONK_{selected}_trainato")


        elif data == "CUP":
            config = utils.load_config_json("../../config_files/config.json" if single_trial else "../../config_files/model_selection_config.json")
            np.random.seed(config["general"]["seed"])

            CUP_train_data = config["paths"]["train_data"]
            CUP_test_data = config["paths"]["test_data"]
            X_train, T_train, input_units = data_loader(CUP_train_data, data_type="train", shuffle=True)
            X_CUP, _, input_units_CUP = data_loader(CUP_test_data, data_type="test", shuffle=False)

            # utils.plot_dataset(X_train, T_train, X_CUP)

            if preprocess == "1":
                X_mean, X_std = utils.standardization(X_train)
                T_mean, T_std = utils.standardization(T_train)
                X_train = (X_train - X_mean) / X_std
                T_train = (T_train - T_mean) / T_std

            elif preprocess == "2":
                X_min, X_max = utils.scaling(X_train)
                T_min, T_max = utils.scaling(T_train)
                X_train = (X_train - X_min) / (X_max - X_min)
                T_train = (T_train - T_min) / (T_max - T_min)
            
            else:
                pass

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

                fig1 = plt.figure(figsize=(5, 4))
                fig2 = plt.figure(figsize=(5, 4))
                nn.plot_metrics(fig1, fig2, title="single_try", data_type="CUP")
            
            else:
                training_sets = [X_train, T_train]
                nn, _, _ = model_assessment(training_sets, input_units, config)
                nn.save_model("../../model_saved/CUP_model.pkl")
                nn.plot("CUP_trainato")

    else:
        members_names = ["Andrea Marcheschi", "Luca Nasini", "Simone Passera"]
        team_name = "ciao team"

        if data == "MONK":
            monk_test_data = f"../../MONK files/monks-{selected}.test"
            X_test, T_test, _ = data_loader(monk_test_data, data_type="MONK")

            nn = utils.neural_network_from_file(f"../../model_saved/MONK_{selected}_model.pkl")

            nn.plot(f"MONK_{selected}_caricato")
            
            T_predicted = np.round(nn.predict(X_test)).astype(int)
            utils.save_predictions(f"../../model_saved/MONK_{selected}_predictions.csv", T_predicted, team_name, members_names)

        elif data == "CUP":
            CUP_train_data = "../../ML-25-PRJ lecture amp package-20251112/ML-CUP25-TR.csv"
            CUP_test_data = "../../ML-25-PRJ lecture amp package-20251112/ML-CUP25-TS.csv"
            X_train, T_train, input_units = data_loader(CUP_train_data, data_type="train", shuffle=True)
            X_CUP, _, input_units_CUP = data_loader(CUP_test_data, data_type="test", shuffle=False)

            if preprocess == 1:
                X_mean, X_std = utils.standardization(X_train)
                T_mean, T_std = utils.standardization(T_train)
                X_train = (X_train - X_mean) / X_std
                T_train = (T_train - T_mean) / T_std
                X_CUP = (X_CUP - X_mean) / X_std

            elif preprocess == 2:
                X_min, X_max = utils.scaling(X_train)
                T_min, T_max = utils.scaling(T_train)
                X_train = (X_train - X_min) / (X_max - X_min)
                T_train = (T_train - T_min) / (T_max - T_min)
                X_CUP = (X_CUP - X_min) / (X_max - X_min)
            
            else:
                pass

            nn = utils.neural_network_from_file("../../model_saved/CUP_model.pkl")

            nn.plot("CUP_caricato")
            
            T_CUP = nn.predict(X_CUP)
            utils.save_predictions("../../model_saved/CUP_predictions.csv", T_CUP, team_name, members_names)

            if preprocess == "1":
                T_CUP = utils.inverse_standardization(T_CUP)

            elif preprocess == "2":
                utils.inverse_scaling(T_CUP, T_min, T_max)
            
            else:
                pass


    end = time.time() - start
    print(f"Elapsed time: {end} s")



    #print(nn)
    #nn.plot()                                   # (da eliminare prima di mandare a Micheli)

    #TODO forse ha senso rimuovere dai config il seme (tanto basta far sì che sia riproducibile con seme hardcodato su macchine diverse, non ci interessa cambiare il seme, oppure vedere se ha senso tenerlo e provare con inizializzazioni diverse)
    #TODO nn validate da fare
    #TODO uniformare impostazione seed randomico da json nei vari file py, non so se sia per come funziona rand ma per ora l'inizializzazione mi sembra essere diversa tra run diverse
    
