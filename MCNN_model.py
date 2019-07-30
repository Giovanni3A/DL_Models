import os, csv, warnings
import numpy as np
import pandas as pd
from PIL import Image
from progressbar import progressbar
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

BASE_PATH  = os.path.dirname(os.path.abspath(__file__))
IMGS_PATH  = os.path.join(BASE_PATH, "Images")
TRAIN_PATH = os.path.join(IMGS_PATH, "train")
TEST_PATH  = os.path.join(IMGS_PATH, "test")

trnfil_pd = pd.read_csv("train.csv")
ntrn      = trnfil_pd.shape[0]
img_dtsz  = len(list(Image.open(os.path.join(TRAIN_PATH,trnfil_pd.filename[0])).convert('LA').getdata(0)))
row_len   = int(img_dtsz**(1/2))

# Multiple y_trains, 1 for each possible
classifications = [np.zeros(shape=(ntrn,1)) for i in range(10)]
X_train = np.zeros(shape=(ntrn, img_dtsz))

print("Loading training images...")
for i in progressbar(range(ntrn)):

    img_file = os.path.join(TRAIN_PATH,trnfil_pd.filename[i])
    label    = trnfil_pd.label[i]
    img_data = np.array(Image.open(img_file).convert('LA').getdata(0))

    X_train[i,:]              = img_data/255
    classifications[label][i] = 1

X_train = X_train.reshape(ntrn,1,row_len,row_len)

def CNN_network(X,y,row_len):
    
    model = Sequential()
    model.add(Conv2D(64, (7,7), input_shape=(1,row_len,row_len), activation='tanh', use_bias=True, kernel_initializer='normal', data_format='channels_first'))
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Conv2D(32, (3,3), activation='tanh', use_bias=True, kernel_initializer='normal', data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(16, activation='tanh', use_bias=True, kernel_initializer='normal'))
    model.add(Dense(1, activation='tanh', use_bias=True, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])

    model.fit(X,y,batch_size=200,epochs=2,verbose=1)

    return model

def classifier_predict(models,X):
    predictions = []
    results     = [model.predict(X) for model in models]
    for ix in range(X.shape[0]):
        ey = [results[m][ix] for m in range(len(models))]
        predictions.append(ey.index(max(ey)))
    return predictions

def classifier_evaluate(models,X,y):
    total_errors = 0
    total_n      = y.shape[0]
    predictions  = classifier_predict(models,X)
    for i in range(len(predictions)):
        if predictions[i] != y[i]:
            total_errors += 1
    return total_errors/total_n

models = [CNN_network(X_train,y_train,row_len) for y_train in classifications]

scores = classifier_evaluate(models, X_train, trnfil_pd.label)
print("Model Error: {}%".format(100*round(scores,3)))