from utils import create_random_extractor, load_config_json
from model import NeuralNetwork
from data_loader import data_loader

import matplotlib.pyplot as plt

import random
import time

if __name__ == "__main__":

    
    start = time.time()

    config = load_config_json("config.json")
    random.seed(config["general"]["seed"])

    train_args = config["training"]

    # MONK DATASET
    monk_train_data = config["paths"]["MONK_train_data"]
    monk_test_data = config["paths"]["MONK_test_data"]
    X_train, t_train, input_units = data_loader(monk_train_data, shuffle=True)              #TODO aggiungere vl splittando ulteriormente x e t train
    X_test, t_test, _ = data_loader(monk_test_data, shuffle=False)


    # # EXAM DATASET
    # CUP_train_data = config["paths"]["CUP_train_data"]
    # X_train, t_train, input_units = data_loader(CUP_train_data, shuffle=False)

    hidden_act_func = config["functions"]["hidden"]
    output_act_func = config["functions"]["output"]
    act_func = [hidden_act_func, output_act_func]                   #TODO mandare al modello nei training hyperpar o in un altro modo
    
    training_hyperpar = config["training"]
    early_stopping = config["training"]["early_stopping"]

    loss_func = config["functions"]["loss"]

    extractor = create_random_extractor(config["initialization"]["method"])

    nn = NeuralNetwork(num_inputs=input_units,
                       num_outputs=config["architecture"]["output_units"],
                       neurons_per_layer=config["architecture"]["neurons_per_layer"],
                       training_hyperpar=training_hyperpar,
                       extractor=extractor,
                       activation=act_func,
                       early_stopping=early_stopping)
    
    # print(nn.feed_forward(one_hot_encoding_train.iloc[0].to_numpy()))
    loss = nn.train(X_train, t_train, train_args, loss_func)

    #TODO nn validate da fare

    nn.test(X_test,t_test)

    end = time.time() - start
    print(end)

    plt.plot(loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
    

    #print(nn)
    #nn.plot()                                   # (da eliminare prima di mandare a Micheli)