import os
import idx2numpy
import numpy as np
import pandas as pd

DATA_PATH  = os.path.dirname(os.path.join(os.curdir,'../data/'))

def read_data(flatten=True):
    print('Loading data...',end='')
    # read training data
    X_train = idx2numpy.convert_from_file(os.path.join(DATA_PATH,'train-images.idx3-ubyte'))
    y_train = idx2numpy.convert_from_file(os.path.join(DATA_PATH,'train-labels.idx1-ubyte'))
    y_train = pd.get_dummies(pd.Series(y_train)).values
    # reshape X matrix into vector
    train_dtsz = X_train.shape[0]
    if flatten:
        X_train = X_train.reshape(train_dtsz,-1)
    else:
        X_train = X_train.reshape(train_dtsz,1,X_train.shape[1],X_train.shape[2])

    # read test data
    X_test  = idx2numpy.convert_from_file(os.path.join(DATA_PATH,'t10k-images.idx3-ubyte'))
    y_test  = idx2numpy.convert_from_file(os.path.join(DATA_PATH,'t10k-labels.idx1-ubyte'))
    y_test = pd.get_dummies(pd.Series(y_test)).values
    # reshape X matrix into vector
    test_dtsz = X_test.shape[0]
    if flatten:
        X_test = X_test.reshape(test_dtsz,-1)
    else:
        X_test = X_test.reshape(test_dtsz,1,X_test.shape[1],X_test.shape[2])
    print('ok')

    return X_train,y_train,X_test,y_test
