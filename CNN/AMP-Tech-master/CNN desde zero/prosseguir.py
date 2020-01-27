import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

longitude, altura = 100, 100
modelo = './modelo/modelo.h5'
pesos = './modelo/pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos)

def predict(file):
  x = load_img(file, target_size=(longitude, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)                           ## pra procesar sem problemas
  array = cnn.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    print("pred: Gato")
  elif answer == 1:
    print("pred: Gorila")
  elif answer == 2:
    print("pred: Cachorro")

  return answer


predict('34.jpg')
