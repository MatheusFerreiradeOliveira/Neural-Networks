import numpy as np
from keras.models import model_from_json

arquivo = open('classificador_breast.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close()

classificador = model_from_json(estrutura_rede)
classificador.load_weights('classificador_breast.h5')

novo = np.array([[15.80, 8.34, 118, 900, 0.10,
                0.25, 0.08, 0.134, 0.1780, 0.20,
                0.05, 1098, 0.87, 4500, 145.8,
                0.005, 0.08, 0.05, 0.015, 0.03,
                0.004, 15.80, 15.80, 175.80, 2018,
                0.14, 0.185, 0.85, 158, 0.365]])

previsao = classificador.predict(novo)
previsao = (previsao > 0.5)

print(previsao)