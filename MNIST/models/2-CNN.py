import os,sys,warnings
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D

sys.path.append('..')
from utils import read_data

warnings.filterwarnings('ignore')
DATA_PATH  = os.path.dirname(os.path.join(os.curdir,'../data/'))

def SequentialNN(dtsz):
    model = Sequential()
    model.add(Conv2D(64, (7,7), input_shape=(1,dtsz[0],dtsz[1]), activation='tanh', use_bias=True, kernel_initializer='normal', data_format='channels_first'))
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Conv2D(32, (3,3), activation='tanh', use_bias=True, kernel_initializer='normal', data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(16, activation='tanh', use_bias=True, kernel_initializer='normal'))
    model.add(Dense(10, activation='softmax', use_bias=True, kernel_initializer='normal'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model

if __name__ == '__main__':
    
    X_train,y_train,X_test,y_test = read_data(flatten=False)

    model = SequentialNN(X_train.shape[-2:])
    model.fit(X_train,y_train,batch_size=200,epochs=10,verbose=1)

    scores = model.evaluate(X_test, y_test, verbose=0)
    print('CNN loss and accuracy:',scores)