"""
Created on Mon Jan 7 14:43:50 2019

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
from keras.models import load_model



test_data = []
with open("imdb_test.txt",  encoding="utf8") as ins:
  for line in ins:
    test_data.append(line)

#print(train_data[2])

temp_X_test = np.array([], dtype='U21')
X_index = 0
X_test = []
for i in range(0,len(test_data)):
  temp = re.findall(r'\S+', test_data[i])
  temp2 = np.append(temp_X_test, temp)
  lut = np.sort(np.unique(temp2))
  ind = np.searchsorted(lut, temp2)
  X_test.append(np.append(temp_X_test, ind))


X_test = keras.preprocessing.sequence.pad_sequences(X_test, padding='post',
                                                        maxlen=100)
model = load_model('trained_model.h5')

prediction = model.predict_classes(X_test, verbose=0)

for i in range (0, len(test_data)):
    print("Prediction of data ", i+1, " is ", prediction[i])

