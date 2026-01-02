import utils
from data_loader import data_loader
from model_selection import model_assessment, launch_trial

import matplotlib.pyplot as plt
import numpy as np

import time
import os
import datetime
import json


if __name__ == "__main__":

    data_input = input("Select which dataset you want to use. (1: MONK; 2: CUP)\n")
    if data_input == "1":
        data = "MONK"
    elif data_input == "2":
        data = "CUP"
    else:
        raise ValueError("Your input must be a number and between 1 and 2")

    if data == "MONK":
        selected = input("Select type of MONK (1, 2, 3).\n")
        if selected not in ["1","2","3"]:
            raise ValueError("Your input must be a number and between 1 and 3")

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

                nn, _, _ = launch_trial(config, [X_train, T_train], [X_val, T_val], input_units, verbose=True)
                
                fig1 = plt.figure(figsize=(5, 4))
                fig2 = plt.figure(figsize=(5, 4))
                save_choice = input("Do you want to save the model? (0: No; 1: Yes)\n")
                if save_choice == "1":
                    dataset_name = config["general"]["dataset_name"]
                    path = f"../../model_saved/{dataset_name}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    os.makedirs(path, exist_ok=True)

                    model_path = os.path.join(path, "model.pkl")
                    nn.save_model(model_path)
                    config_path = os.path.join(path, "config.json")
                    with open(config_path, "w") as file:
                        json.dump(config, file)
                    result_path = os.path.join(path, "result.txt")
                    with open(result_path, "w") as file:
                        file.write(f"Date:\t\t\t\t\t{datetime.datetime.now().isoformat()}\n")
                        file.write(f"Validation Loss:\t\t{nn.best_loss:.6f}\n")
                        if nn.best_accuracy is not None:
                            file.write(f"Validation Accuracy:\t{nn.best_accuracy:.2%}\n")
                        file.write(f"Best Epoch:\t\t\t{nn.best_epoch}")

                    nn.plot_metrics(fig1, fig2, title="single_trial", data_type=dataset_name, save_path=path)
                else:
                    nn.plot_metrics(fig1, fig2, title="single_trial", data_type=config["general"]["dataset_name"])
            else:
                training_sets = [X_train, T_train]
                test_sets = [X_test, T_test]
                nn, _, _ = model_assessment(training_sets,input_units, config, test_sets)

        elif data == "CUP":
            config = utils.load_config_json("../../config_files/config.json" if single_trial else "../../config_files/model_selection_config.json")
            np.random.seed(config["general"]["seed"])

            CUP_train_data = config["paths"]["train_data"]
            CUP_test_data = config["paths"]["test_data"]
            X_train, T_train, input_units = data_loader(CUP_train_data, data_type="train", shuffle=True)
            X_CUP, _, input_units_CUP = data_loader(CUP_test_data, data_type="test", shuffle=False)

            if single_trial:

                data_split_prop = [config["training"]["splitting"]["tr"], config["training"]["splitting"]["vl"], config["training"]["splitting"]["ts"]]
                X_train, X_val, X_test, T_train, T_val, T_test = utils.data_splitting(X_train, T_train, data_split_prop)

                nn, _, _ = launch_trial(config, [X_train, T_train], [X_val, T_val], input_units, verbose=True)

                fig1 = plt.figure(figsize=(5, 4))
                fig2 = plt.figure(figsize=(5, 4))
                save_choice = input("Do you want to save the model? (0: No; 1: Yes)\n")
                if save_choice == "1":
                    dataset_name = config["general"]["dataset_name"]
                    path = f"../../model_saved/{dataset_name}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    os.makedirs(path, exist_ok=True)

                    model_path = os.path.join(path, "model.pkl")
                    nn.save_model(model_path)
                    config_path = os.path.join(path, "config.json")
                    with open(config_path, "w") as file:
                        json.dump(config, file)
                    result_path = os.path.join(path, "result.txt")
                    with open(result_path, "w") as file:
                        file.write(f"Date:\t\t\t\t\t{datetime.datetime.now().isoformat()}\n")
                        file.write(f"Validation Loss:\t\t{nn.best_loss:.6f}\n")
                        if nn.best_accuracy is not None:
                            file.write(f"Validation Accuracy:\t{nn.best_accuracy:.2%}\n")
                        file.write(f"Best Epoch:\t\t\t{nn.best_epoch}")

                    nn.plot_metrics(fig1, fig2, title="single_trial", data_type=dataset_name, save_path=path)
                else:
                    nn.plot_metrics(fig1, fig2, title="single_trial", data_type=config["general"]["dataset_name"])
            
            else:
                training_sets = [X_train, T_train]
                nn, _, _ = model_assessment(training_sets, input_units, config)

    else:
        members_names = ["Andrea Marcheschi", "Luca Nasini", "Simone Passera"]
        team_name = "ciao team"

        if data == "MONK":
            monk_test_data = f"../../MONK files/monks-{selected}.test"
            X_test, T_test, _ = data_loader(monk_test_data, data_type="MONK")

            config = utils.load_config_json(f"../../config_files/MONK_{selected}_config.json")
            dataset_name = config["general"]["dataset_name"]

            base_path = f"../../model_saved/{dataset_name}"

            if not os.path.exists(base_path):
                raise FileNotFoundError(f"Directory {base_path} not found")

            timestamps = sorted(os.listdir(base_path))
            
            if not timestamps:
                raise FileNotFoundError(f"No models in {base_path}")
                
            latest_run = timestamps[-1]
            model_path = os.path.join(base_path, latest_run, "model.pkl")

            nn = utils.neural_network_from_file(model_path)

            T_predicted = np.round(nn.predict(X_test)).astype(int)
            utils.save_predictions(f"../../model_saved/MONK_{selected}_predictions.csv", T_predicted, team_name, members_names)

        elif data == "CUP":
            CUP_train_data = "../../ML-25-PRJ lecture amp package-20251112/ML-CUP25-TR.csv"
            CUP_test_data = "../../ML-25-PRJ lecture amp package-20251112/ML-CUP25-TS.csv"
            X_train, T_train, input_units = data_loader(CUP_train_data, data_type="train", shuffle=True)
            X_CUP, _, input_units_CUP = data_loader(CUP_test_data, data_type="test", shuffle=False)

            config = utils.load_config_json(f"../../config_files/config.json")
            dataset_name = config["general"]["dataset_name"]

            base_path = f"../../model_saved/{dataset_name}"

            if not os.path.exists(base_path):
                raise FileNotFoundError(f"Directory {base_path} not found")

            timestamps = sorted(os.listdir(base_path))
            
            if not timestamps:
                raise FileNotFoundError(f"No models in {base_path}")
                
            latest_run = timestamps[-1]
            model_path = os.path.join(base_path, latest_run, "model.pkl")
            
            nn = utils.neural_network_from_file(model_path)

            preprocess = nn.preprocessing
            X_params = nn.X_params
            T_params = nn.T_params
            
            if preprocess == "standardization":
                X_CUP = (X_CUP - X_params[0]) / X_params[1]

            elif preprocess == "rescaling":
                X_CUP = (X_CUP - X_params[0]) / (X_params[1] - X_params[0])
            
            else:
                pass            

            T_CUP = nn.predict(X_CUP)

            if preprocess == "standardization":
                T_CUP_real = utils.inverse_standardization(T_CUP, *T_params)

            elif preprocess == "rescaling":
                T_CUP_real = utils.inverse_scaling(T_CUP, *T_params)
            
            else:
                T_CUP_real = T_CUP

            utils.save_predictions("../../model_saved/CUP_predictions.csv", T_CUP_real, team_name, members_names)


    end = time.time() - start
    print(f"Elapsed time: {end} s")
