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

y_train = np.zeros(shape=(ntrn,10))
X_train = np.zeros(shape=(ntrn, img_dtsz))

print("Loading training images...")
for i in progressbar(range(ntrn)):

    img_file = os.path.join(TRAIN_PATH,trnfil_pd.filename[i])
    label    = trnfil_pd.label[i]
    img_data = np.array(Image.open(img_file).convert('LA').getdata(0))

    X_train[i,:]     = img_data/255
    y_train[i,label] = 1

X_train = X_train.reshape(ntrn,1,row_len,row_len)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=666)

def CNN_network(X,y,row_len):
    model = Sequential()
    model.add(Conv2D(64, (7,7), input_shape=(1,row_len,row_len), activation='tanh', use_bias=True, kernel_initializer='normal', data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32, (3,3), activation='tanh', use_bias=True, kernel_initializer='normal', data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(16, activation='tanh', use_bias=True, kernel_initializer='normal'))
    model.add(Dense(10, activation='softmax', use_bias=True, kernel_initializer='normal'))

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    model.fit(X,y,batch_size=200,epochs=10,verbose=1)
    return model

model = CNN_network(X_train,y_train,row_len)

scores = model.evaluate(X_test, y_test, verbose=0)
print("MLP Model Error: {}".format(round(100-scores[1]*100),2))