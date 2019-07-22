import os, csv, progressbar, warnings
import numpy as np
import pandas as pd
from PIL import Image
from progressbar import progressbar
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.utils import np_utils
from keras.datasets import mnist

warnings.filterwarnings("ignore")

BASE_PATH  = os.path.dirname(os.path.abspath(__file__))
IMGS_PATH  = os.path.join(BASE_PATH, "Images")
TRAIN_PATH = os.path.join(IMGS_PATH, "train")
TEST_PATH  = os.path.join(IMGS_PATH, "test")

trnfil_pd = pd.read_csv("train.csv")
ntrn      = trnfil_pd.shape[0]
img_dtsz  = len(list(Image.open(os.path.join(TRAIN_PATH,trnfil_pd.filename[0])).convert('LA').getdata(0)))

y_train = np.zeros(shape=(ntrn,10))
X_train = np.zeros(shape=(ntrn,img_dtsz))

print("Loading training images...")
for i in progressbar(range(ntrn)):

    img_file = os.path.join(TRAIN_PATH,trnfil_pd.filename[i])
    label    = trnfil_pd.label[i]
    img_data = np.array(Image.open(img_file).convert('LA').getdata(0))

    X_train[i,:]     = img_data/255
    y_train[i,label] = 1

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=666)

def MLP_network(X,y,dtsz,ntrn):
    model = Sequential()
    model.add(Dense(dtsz, kernel_initializer='normal', activation='tanh', input_dim=img_dtsz,use_bias=True))
    model.add(Dense(10, kernel_initializer='normal', activation='softmax',use_bias=True))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    model.fit(X,y,batch_size=200,epochs=10,verbose=1)
    return model

def validation_MLP_network(X,y,dtsz,ntrn):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=33)

    model = Sequential()
    model.add(Dense(dtsz, kernel_initializer='normal', activation='tanh', input_dim=img_dtsz)) 
    model.add(Dense(10, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    model.fit(X_train,y_train,validation_data=(X_valid, y_valid),batch_size=200,epochs=10,verbose=1)
    return model

model = MLP_network(X_train,y_train,img_dtsz,ntrn)

scores = model.evaluate(X_test, y_test, verbose=0)
print("MLP Model Error: {}".format(round(100-scores[1]*100),2))