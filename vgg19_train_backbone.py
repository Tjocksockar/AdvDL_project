#from random import shuffle
import tensorflow as tf
import keras.backend as K
#import copy
#import scipy
#import numpy as np
#import os
#from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG19
#from keras.losses import categorical_crossentropy, kullback_leibler_divergence
from keras.layers import Dense, Flatten, Input, Conv2D, BatchNormalization, Activation, Conv2DTranspose, Multiply, Dropout
from keras.models import Model, Sequential, load_model
from keras.activations import relu, sigmoid
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint

from data_generator import *

def build_transfer_model():
  model = VGG19(include_top=False, weights='imagenet',input_shape=(224,224,3))
  input_shape = model.layers[0].output_shape[1:3]
  print(input_shape)
  transfer_layer = model.get_layer('block5_pool')

  conv_model = Model(inputs=model.input, outputs=transfer_layer.output)

  new_model = Sequential()
  new_model.add(conv_model)
  new_model.add(Flatten())
  new_model.add(Dense(1024,activation='relu'))
  new_model.add(Dropout(0.25))
  new_model.add(Dense(100,activation='softmax'))
  conv_model.trainable = False

  return input_shape, new_model

if __name__ == '__main__':
  filename = "checkpoint.hdf5"
  first_time=True
  dataset = "cifar100"
  if first_time:
    input_shape, new_model = build_transfer_model()
  else:
    input_shape=(224,224)
    filename="checkpoint_"+dataset+".hdf5"
    new_model = load_model(filename)
  print()
  print(new_model.layers[0].layers[0].output.shape)
  print()
  batch_size = 64
  epochs = 1024
  steps_per_epoch = 200

  train_list, val_list = create_generator_input()
  steps_per_epoch = int(len(train_list)/batch_size)
  val_steps_per_epoch = int(len(val_list)/batch_size)

  if first_time:
      optimizer = Adam()
  else:
      optimizer = Adam(lr=10**-5)
  loss = 'categorical_crossentropy'
  metrics = ['categorical_accuracy']
  checkpoint = ModelCheckpoint(filename, monitor='loss', verbose=0, save_best_only=True, mode='min')
  callbacks = [checkpoint]

  new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
  new_model.fit_generator(generator=generator_backbone(train_list),
                          epochs=epochs,
                          steps_per_epoch = steps_per_epoch,
                          validation_data=generator_backbone(val_list),
                          validation_steps=val_steps_per_epoch,
                          callbacks=callbacks)

