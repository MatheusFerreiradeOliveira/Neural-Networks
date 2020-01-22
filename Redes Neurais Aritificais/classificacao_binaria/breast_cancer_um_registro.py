import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

classificador = Sequential()

classificador.add(Dense(units = 8, activation = 'relu', kernel_initializer = 'normal', input_dim = 30))
#zera 20% das entradas, dropout previne overfitting
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 8, activation = 'relu', kernel_initializer = 'normal'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 1, activation = 'sigmoid'))
otimizador = keras.optimizers.Adam(lr=0.001, decay=0.0001, clipvalue=0.5)
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

classificador.fit(previsores, classe, batch_size=10, epochs=100)

novo = np.array([[15.80, 8.34, 118, 900, 0.10,
                0.25, 0.08, 0.134, 0.1780, 0.20,
                0.05, 1098, 0.87, 4500, 145.8,
                0.005, 0.08, 0.05, 0.015, 0.03,
                0.004, 15.80, 15.80, 175.80, 2018,
                0.14, 0.185, 0.85, 158, 0.365]])

previsao = classificador.predict(novo)
previsao = (previsao > 0.5)

print(previsao)