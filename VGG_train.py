import tensorflow as tf
import keras.backend as K
import copy
import scipy
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.losses import categorical_crossentropy, kullback_leibler_divergence
from keras.layers import Dense, Flatten, Input, Conv2D, BatchNormalization, Activation, Conv2DTranspose, Multiply, Dropout
from keras.models import Model, Sequential
from keras.activations import relu, sigmoid
from keras.optimizers import Adam, RMSprop
from keras.datasets import cifar100

def build_transfer_model(): 
  model = VGG16(include_top=False, weights='imagenet',input_shape=(224,224,3))
  input_shape = model.layers[0].output_shape[1:3]
  print(input_shape)
  transfer_layer = model.get_layer('block5_pool')

  conv_model = Model(inputs=model.input, outputs=transfer_layer.output)

  new_model = Sequential()
  new_model.add(conv_model)
  new_model.add(Flatten())
  new_model.add(Dense(1024,activation='relu'))
  new_model.add(Dropout(0.25))
  new_model.add(Dense(200,activation='softmax'))
  return new_model


if __name__ = '__main__':
  new_model = build_transfer_model()
  batch_size = 64

  datagen_train = ImageDataGenerator(rescale=1/255)
  datagen_test =ImageDataGenerator(rescale=1/255)

  generator_train = datagen_train.flow_from_directory(directory='pics/train',
                                                      target_size = input_shape,
                                                      batch_size=batch_size,
                                                      shuffle = True)
  #
  generator_test = datagen_train.flow_from_directory(directory='pics/test',
                                                     target_size=input_shape,
                                                     batch_size=batch_size,
                                                     shuffle=False)

  steps_test = generator_test.n / batch_size



  transfer_layer = model.get_layer('block5_pool')

  conv_model = Model(inputs=model.input, outputs=transfer_layer.output)

  new_model = Sequential()
  new_model.add(conv_model)
  new_model.add(Flatten())
  new_model.add(Dense(1024,activation='relu'))
  new_model.add(Dropout(0.5))
  new_model.add(Dense(100,activation='softmax'))

  optimizer = Adam(lr=1e-5)
  loss = 'categorical_crossentropy'
  metrics = ['categorical_accuracy']

  conv_model.trainable = False
  for layer in conv_model.layers:
      layer.trainable = False

  new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

  epochs = 20
  steps_per_epoch = 100

  # history = new_model.fit_generator(generator=generator_train,epochs=epochs,
  # steps_per_epoch = steps_per_epoch,validation_data=generator_test,validation_steps =steps_test)
  new_model.summary()

  #new_model.fit(x_train_new,y_train,batch_size=batch_size,epochs=epochs,validation_data=(x_test_new,y_test))
  new_model.fit_generator(generator=generator_train,
                          epochs=epochs,
                          steps_per_epoch = steps_per_epoch,
                          validation_data=generator_test,
                          validation_steps=steps_test)
