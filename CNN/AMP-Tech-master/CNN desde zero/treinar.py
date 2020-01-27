import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K

K.clear_session()

data_emtreinamento='./data/emtreinamento'
data_validacao='./data/validacao'

## Parametros para as imagens

epocas=20                                                                ##quantidade de vezes de que vai interar
altura, longitude= 100, 100
batch_size= 32                                                                 ## Processamento de cada passo
passos= 1000
passos_validacao=200
filtrosConvu1=32                                                                    ##profundidade
filtrosConvu2=64
tamanho_filtro1=(3,3)
tamanho_filtro2=(2,2)
tamanho_pool=(2,2)                                                                  ##tamanho do filtro utilizado no MaxPooling
classes=3
lr=0.0005                                                                       ##ajustes para dar soluções novas

#pre processamento de imagens

emtreinamento_datagen= ImageDataGenerator(
    rescale=1./255,                                                             ##redimensiona a imagem
    shear_range=0.3,                                                                ##inclinar
    zoom_range=0.3,
    horizontal_flip=True                                                              ## inverter
)
validacao_datagen=ImageDataGenerator(
    rescale=1./255                                                                       ##não precisa invertar
)

imagem_emtreinamento=emtreinamento_datagen.flow_from_directory(
    data_emtreinamento,
    target_size=(altura, longitude),
    batch_size=batch_size,
    class_mode='categorical'
)

imagem_validacao=validacao_datagen.flow_from_directory(
    data_validacao,
    target_size=(altura, longitude),
    batch_size=batch_size,
    class_mode='categorical'
)

## rede cnn

cnn=Sequential()
cnn.add(Convolution2D(filtrosConvu1, tamanho_filtro1, padding='same', input_shape=(altura, longitude,3), activation='relu'))

cnn.add(MaxPooling2D(pool_size=tamanho_pool))

cnn.add(Convolution2D(filtrosConvu2, tamanho_filtro2, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamanho_pool))

cnn.add(Flatten())                                                                            ##imagem  com uma dimensão
cnn.add(Dense(256, activation='relu'))                                                      ##todos os neuronios estão conectados com a anterior
cnn.add(Dropout(0.5))                                                                           ##apagar 50% dos neuronios cada passo
cnn.add(Dense(classes, activation='softmax'))                                                   ##softmax probabilidade

cnn.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=lr), metrics=['accuracy'])                                ##otimizar

cnn.fit(imagem_emtreinamento, steps_per_epoch=passos, epochs=epocas, validation_data=imagem_validacao, validation_steps=passos_validacao)           ##treinar a rede

dir ='./modelo'

if not os.path.exists(dir):
    os.mkdir(dir)
cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')
