from random import shuffle
import matplotlib.image as im
import tensorflow as tf
import keras.backend as K
import copy
import scipy
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16, ResNet50, ResNet101, ResNet152
from keras.losses import categorical_crossentropy, kullback_leibler_divergence
from keras.layers import Dense, Flatten, Input, Conv2D, BatchNormalization, Activation, Conv2DTranspose, Multiply, Dropout
from keras.models import Model, Sequential, load_model
from keras.activations import relu, sigmoid
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint
from skimage.io import imread
from skimage.transform import rescale, resize
from vgg16_train_backbone import *
from model import *
from data_generator import *

if __name__ == '__main__':
  first_time=True
  dataset_filename = 'tiny-imagenet-200'
  train_list, val_list = create_generator_input_imagenet(dataset_filename=dataset_filename)

  TIN = "TinyImageNet"
  CIFAR = "CIFAR"
  dataset = TIN
  if dataset==TIN:
    classes = 200
  else:
    classes = 100
  version=152
  filename="checkpoint_"+dataset+"ResNet"+str(version)+".hdf5"
  if first_time:
    input_shape, new_model = build_resnet_model(classes=classes, version=version, )
  else:
    input_shape=(224,224)
    new_model = load_model(filename)
  print(new_model.layers[0].layers[0].output.shape)
  batch_size = 64
  epochs = 1024

  steps_test = len(val_list) // batch_size #number of validation steps
  steps_train = len(train_list) // batch_size
  if first_time:
      optimizer = Adam()
  else:
      optimizer = Adam(lr=10**-5)
  loss = 'categorical_crossentropy'
  metrics = ['categorical_accuracy']
  checkpoint = ModelCheckpoint(filename, monitor='loss', verbose=0, save_best_only=True, mode='min')
  callbacks = [checkpoint]

  new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
  new_model.fit_generator(generator=generator_backbone_imagenet(train_list),
                          epochs=epochs,
                          steps_per_epoch = steps_train,
                          validation_data=generator_backbone_imagenet(val_list),
                          validation_steps=steps_test,
                          callbacks=callbacks)

