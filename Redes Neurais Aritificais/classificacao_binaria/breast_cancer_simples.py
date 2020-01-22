import pandas as pd

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

from sklearn.model_selection import train_test_split

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)

import keras
from keras.models import Sequential
from keras.layers import Dense

classificador = Sequential()

#units passa o somatório das (entradas + saidas)/2, no caso são 30 atrib de rntrada e 1 de saida, então da 31/2 que da 16,
#no segundo é a funcao la, no terceiro é o tamanho da entrada (30 atributos) e só usa o tereiro na primeira camada
classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform', input_dim = 30))
classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'))

#usar o Dense significa q a rede vai ser Fully conected xD
classificador.add(Dense(units = 1, activation = 'sigmoid'))

#lr é a learning rate, decay é o tanto q a LR cai por geração e clipvalue é para manter o bixo dentro do escopo (-x, x) -> (-0.5 até 0.5)
otimizador = keras.optimizers.Adam(lr=0.01, decay=0.001, clipvalue=0.5)

#adam = descida do gradiente estocastico
classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

#fit é de encaixar, batch_size é q vai pegar 10 registros, dps +10 e assim vai (para atualizar os pesos (MINI BATCH))
classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10, epochs = 100)

#vizualização dos pesos
pesos0 = classificador.layers[0].get_weights()
print(pesos0)
pesos1 = classificador.layers[1].get_weights()
print(pesos1)
pesos2 = classificador.layers[2].get_weights()
print(pesos2)

previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)

#Um jeito de fazer
from sklearn.metrics import confusion_matrix, accuracy_score

precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

print("precisão com SKLEARN:", precisao)
print(matriz)

#Outro jeito de fazer
resultado = classificador.evaluate(previsores_teste, classe_teste)

#print("precisao com o keras:", resultado)
print("Perda com o keras:", resultado[0])
print("Acerto com o keras:", resultado[1])


