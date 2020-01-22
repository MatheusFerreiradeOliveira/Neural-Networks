import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

def criarRede():
    classificador = Sequential()

    classificador.add(Dense(units = 8, activation = 'relu', kernel_initializer = 'normal', input_dim = 30))
    #zera 20% das entradas, dropout previne overfitting
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 8, activation = 'relu', kernel_initializer = 'normal'))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 8, activation = 'relu', kernel_initializer = 'normal'))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 8, activation = 'relu', kernel_initializer = 'normal'))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 1, activation = 'sigmoid'))
    otimizador = keras.optimizers.Adam(lr=0.001, decay=0.0001, clipvalue=0.5)
    classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn= criarRede,
                                epochs = 100,
                                batch_size = 10)

#X são os previsores, Y = a classe
resultados = cross_val_score(estimator = classificador, X = previsores, y = classe, cv = 10, scoring='accuracy')

media = resultados.mean()
print(media)

#Desvio padrão, quanto maior mais perto do overfitting (Ta se acostumando demais naquela base de dados) 
desvio = resultados.std()
print(desvio)