import keras
from keras.layers import Conv2D
from keras.models import Sequential

model = Sequential()
model.add(Conv2D(3, (3, 3), strides=(2, 2), padding='same', input_shape=(32, 32, 3)))

model.summary()
