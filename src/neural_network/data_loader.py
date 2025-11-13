import pandas as pd

import random

def preprocess_monk(data, type):
    """ Preprocess the MONK dataset by performing one-hot encoding on categorical features.  Returns the one-hot encoded patterns, targets and the number of input units."""
    if type == "train":
        data = pd.read_csv(data, sep=r"\s+", header=None)
        # print(data.head())
    elif type == "test":
        data = pd.read_csv(data, sep=r"\s+", header=None)
        # print(data.head())
    else:
        raise ValueError("Invalid type. Must be 'train' or 'test'.")

    # the first column in the dataset is the target, the remaining values are the pattern
    targets = data[data.columns[0]]
    # print(targets.head(25))

    # removing target and name
    patterns = data.drop(data.columns[0], axis=1)
    patterns = patterns.drop(data.columns[-1], axis=1)
    # print(patterns.head(5))

    # this is necessary for pandas to get the dummies otherwise it isn't happy
    patterns = patterns.astype(str)

    one_hot_encoding = pd.get_dummies(patterns, prefix=['A1','A2','A3','A4','A5','A6'])
    # print(one_hot_encoding)

    input_units_number = one_hot_encoding.shape[1]
    return one_hot_encoding,targets,input_units_number

def preprocess_exam_file(data):
    """ Preprocess the exam dataset.  Returns the targets and the number of input units."""
    data = pd.read_csv(data, sep=r"\s+", header=None)

    # the lasts 4 columns in the dataset are the targets, the remaining values are the pattern and examples index
    targets = data[data.columns[-4:]]
    # print(targets.head(25))

    #removing target and name
    patterns = data.drop(data.columns[-4:], axis=1)
    patterns = patterns.drop(data.columns[0], axis=1)
    # print(patterns.head(5))

    input_units_number = patterns.shape[1]
    return targets,input_units_number