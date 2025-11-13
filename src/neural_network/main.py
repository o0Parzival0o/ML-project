from utils import create_random_extractor, load_config_json
from model import NeuralNetwork
from data_loader import preprocess_monk, preprocess_exam_file

import random

if __name__ == "__main__":

    config = load_config_json("config.json")
    random.seed(config["general"]["seed"])

    # MONK DATASET
    monk_train_data = config["paths"]["MONK_train_data"]
    monk_test_data = config["paths"]["MONK_test_data"]
    one_hot_encoding_train,targets_train,input_units_number = preprocess_monk(monk_train_data, "train")
    one_hot_encoding_test,targets_test,_ = preprocess_monk(monk_test_data, "test")

    # # EXAM DATASET
    # CUP_train_data = config["paths"]["CUP_train_data"]
    # one_hot_encoding, targets, input_units_number = preprocess_exam_file()

    hidden_act_func = config["functions"]["hidden"]
    output_act_func = config["functions"]["output"]
    act_func = [hidden_act_func, output_act_func]
    
    training_hyperpar = [config["training"]["learning_rate"], config["training"]["momentum"], config["training"]["batch_size"]]

    extractor = create_random_extractor(config["initialization"]["method"])

    nn = NeuralNetwork(num_inputs=input_units_number,
                       num_outputs=config["architecture"]["output_units"],
                       neurons_per_layer=config["architecture"]["neurons_per_layer"],
                       training_hyperpar=training_hyperpar,
                       extractor=extractor)

    print(nn)
    nn.plot()                                   # (da eliminare prima di mandare a Micheli)