import pandas as pd
import numpy as np

def preprocess_monk(path_file):
    """ Preprocess the MONK dataset by performing one-hot encoding on categorical features.  Returns the one-hot encoded patterns, targets and the number of input units."""
    if "train" in path_file:
        data = pd.read_csv(path_file, sep=r"\s+", header=None)
    elif "test" in path_file:
        data = pd.read_csv(path_file, sep=r"\s+", header=None)
    else:
        raise ValueError("Invalid type. Must be 'train' or 'test'.")

    # the first column in the dataset is the target, the remaining values are the pattern
    targets = data[data.columns[0]]
    # print(targets.head(25))

    # removing target and name
    patterns = data.drop(data.columns[0], axis=1)
    patterns = patterns.drop(data.columns[-1], axis=1)

    # this is necessary for pandas to get the dummies otherwise it isn't happy
    patterns = patterns.astype(str)

    one_hot_encoding = pd.get_dummies(patterns, prefix=['A1','A2','A3','A4','A5','A6'])
    # print(one_hot_encoding)

    input_units_number = one_hot_encoding.shape[1]
    return one_hot_encoding, targets, input_units_number

def preprocess_exam_file(file_path):
    """ Preprocess the exam dataset.  Returns the targets and the number of input units."""
    data = pd.read_csv(file_path, sep=",", header=None, comment='#')

    # the lasts 4 columns in the dataset are the targets, the remaining values are the pattern and examples index
    targets = data[data.columns[-4:]]

    X = data[data.columns[1:-4]]            # delete idx column

    #removing target and name
    patterns = data.drop(data.columns[-4:], axis=1)
    patterns = patterns.drop(data.columns[0], axis=1)

    input_units_number = X.shape[1]
    return X, targets, input_units_number

def data_loader(file_path, shuffle=False):
    if "monks" in file_path:
        X, t, input_units = preprocess_monk(file_path)
    else:
        X, t, input_units = preprocess_exam_file(file_path)

    X = X.to_numpy()
    t = t.to_numpy()
    idx = np.arange((len(X)))

    if shuffle:
        np.random.shuffle(idx)
        X = X[idx]
        t = t[idx]

    return X, t, input_units


    
