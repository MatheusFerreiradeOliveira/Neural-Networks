import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder #para tratar as strings para numeros
from keras.utils import np_utils #para ajeitar a base de treinamento (colocar a quantidade de colunas igual a de saidas possiveis)

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

labelencode = LabelEncoder()

classe = labelencode.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe_dummy, test_size= 0.25)

classificador = Sequential()
#units = 4 atrib + 3 classes / 2 = 4, input data = qtd atrib
classificador.add(Dense(units=4, activation= 'relu', input_dim=4))
classificador.add(Dense(units=4, activation= 'relu'))
classificador.add(Dense(units=3, activation= 'softmax'))

classificador.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10, epochs = 1000)

#um jeito
resultado = classificador.evaluate(previsores_teste, classe_teste)

#outro jeito
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)

#agora precisa colocar a classe em uma unica coluna para fzr a matriz
import numpy as np
classe_teste2 = [np.argmax(t) for t in classe_teste]
previsoes2 = [np.argmax(t) for t in previsoes]

from sklearn.metrics import confusion_matrix
matriz = confusion_matrix(previsoes2, classe_teste2)

print(matriz)
