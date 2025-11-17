from utils import create_random_extractor, load_config_json
from model import NeuralNetwork
from data_loader import data_loader

import random

if __name__ == "__main__":

    

    config = load_config_json("config.json")
    random.seed(config["general"]["seed"])

    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]

    # MONK DATASET
    monk_train_data = config["paths"]["MONK_train_data"]
    monk_test_data = config["paths"]["MONK_test_data"]
    X_train, t_train, input_units = data_loader(monk_train_data, shuffle=True)
    X_test, t_test, _ = data_loader(monk_train_data, batch_size=batch_size, shuffle=False)


    # # EXAM DATASET
    # CUP_train_data = config["paths"]["CUP_train_data"]
    # X_train, t_train, input_units = data_loader(CUP_train_data, shuffle=False)

    hidden_act_func = config["functions"]["hidden"]
    output_act_func = config["functions"]["output"]
    act_func = [hidden_act_func, output_act_func]
    
    training_hyperpar = [config["training"]["learning_rate"], config["training"]["momentum"], config["training"]["batch_size"]]

    extractor = create_random_extractor(config["initialization"]["method"])

    nn = NeuralNetwork(num_inputs=input_units,
                       num_outputs=config["architecture"]["output_units"],
                       neurons_per_layer=config["architecture"]["neurons_per_layer"],
                       training_hyperpar=training_hyperpar,
                       extractor=extractor)
    
    # print(nn.feed_forward(one_hot_encoding_train.iloc[0].to_numpy()))
    nn.train(X_train, t_train,epochs,batch_size)

    #print(nn)
    #nn.plot()                                   # (da eliminare prima di mandare a Micheli)