# -*- coding: utf-8 -*-
"""
Created on Mon Jan 7 13:23:54 2019

@author: Sefira Karina
"""
import pandas
import re
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.core import Activation

ytrain_data = pandas.read_csv('imdb_train.txt', header=None, engine='python')
y_train = ytrain_data.apply(lambda x: x.str.split().str[0])

train_data = []
with open("imdb_train.txt",  encoding="utf8") as ins:
  for line in ins:
    train_data.append(line)

#print(train_data[2])

temp_X_train = np.array([], dtype='U21')
X_index = 0
X_train = []

#convert string to number
for i in range(0,len(train_data)):
  temp = re.findall(r'\S+', train_data[i])
  temp2 = np.append(temp_X_train, temp)
  lut = np.sort(np.unique(temp2))
  ind = np.searchsorted(lut, temp2)
  X_train.append(np.append(temp_X_train, ind))


#make all input lenght the same
X_train = keras.preprocessing.sequence.pad_sequences(X_train, padding='post',
                                                        maxlen=100)

input_dim = X_train.shape[1]


model = Sequential()

model.add(Dense(100, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(30, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(np.array(X_train),np.array(y_train), epochs=150)

model.save('trained_model.h5')

'''''
train_data = []
with open("imdb_train.txt",  encoding="utf8") as ins:
  for line in ins:
    train_data.append(line)

  global train_data
with open('imdb_train.txt',  encoding="utf8") as f:
  train_data = f.readlines()  

  train_data = pandas.read_csv('imdb_train.txt', header=None, delimiter="\t", engine='python')

'''
'''''
train_data = pandas.read_csv('imdb_train.txt', header=None, engine='python')

#y_train = (train_data.iloc[:,1:].values).astype('float32')
#y_train =  train_data.iloc[:,0].values.astype('int32')

y_train = train_data.apply(lambda x: x.str.split().str[0])

print(train_data[2])

X_train = [[]]
X_index = 0

for i in range(0,len(train_data)):
  temp = re.findall(r'\S+', train_data[i].apply(str))
  X_train.append(temp)
'''