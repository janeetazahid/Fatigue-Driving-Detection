# -*- coding: utf-8 -*-
"""
Trains the LSTM model, trained on Google Colab 
"""
#IMPORTS
import numpy as np
import pandas as pd
import cv2
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers import Dropout
# %tensorflow_version 2.x
from tensorflow import keras
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split


#import dataset 
df=pd.read_excel('LSTM_200.xlsx')

#display dataset shape
df.shape

#extract values 
values = df.values
values=values.astype('float32')

#extract features 
X=values[:,:-3]
#extract target variables 
Y=values[:,400:]
#train test split 
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=4)

#reshape for LSTM model
x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
x_test= x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
print(x_train.shape, y_train.shape,  x_test.shape, y_test.shape)

#create model 
model = Sequential()
model.add(LSTM(300, input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# implement early stopping and train model 
es = keras.callbacks.EarlyStopping(monitor='accuracy', mode='max', verbose=1,patience=25,restore_best_weights=True)
model.fit(x_train, y_train, epochs=4000,batch_size=32,callbacks=[es])
#save model 
model.save('model_dense300')
#evaluate on the test set 
model.evaluate(x_test, y_test)




