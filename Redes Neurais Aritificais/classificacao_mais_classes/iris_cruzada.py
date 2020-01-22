import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder #para tratar as strings para numeros
from keras.utils import np_utils #para ajeitar a base de treinamento (colocar a quantidade de colunas igual a de saidas possiveis)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values
labelencode = LabelEncoder()
classe = labelencode.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

def criar_rede():
    classificador = Sequential()
    #units = 4 atrib + 3 classes / 2 = 4, input data = qtd atrib
    classificador.add(Dense(units=4, activation= 'relu', input_dim=4))
    classificador.add(Dense(units=4, activation= 'relu'))
    classificador.add(Dense(units=3, activation= 'softmax'))
    classificador.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    return classificador

print(matriz)
