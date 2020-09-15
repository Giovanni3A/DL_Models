import os,sys,warnings
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

sys.path.append('..')
from utils import read_data

warnings.filterwarnings('ignore')
DATA_PATH  = os.path.dirname(os.path.join(os.curdir,'../data/'))

def SequentialNN(dtsz):
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=dtsz,use_bias=True))
    model.add(Dense(10, activation='softmax',use_bias=True))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

if __name__ == '__main__':
    
    X_train,y_train,X_test,y_test = read_data()

    model = SequentialNN(X_train.shape[1])
    model.fit(X_train,y_train,batch_size=100,epochs=10,verbose=1)

    scores = model.evaluate(X_test, y_test, verbose=0)
    print('Sequential model loss and accuracy:',scores)