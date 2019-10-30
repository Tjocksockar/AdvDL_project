from random import shuffle
import matplotlib.image as im
import tensorflow as tf
import keras.backend as K
import copy
import scipy
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.losses import categorical_crossentropy, kullback_leibler_divergence
from keras.layers import Dense, Flatten, Input, Conv2D, BatchNormalization, Activation, Conv2DTranspose, Multiply, Dropout
from keras.models import Model, Sequential, load_model
from keras.activations import relu, sigmoid
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint
from skimage.io import imread
from skimage.transform import rescale, resize

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
  new_model.add(Dense(100,activation='softmax'))
  conv_model.trainable = False

  return input_shape, new_model



def generator(samples, batch_size=64,shuffle_data=True):
    """
    Yields the next training batch.
    Suppose `samples` is an array [[image1_filename,label1], [image2_filename,label2],...].
    """
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        shuffle(samples)

        # Get index to start each batch: [0, batch_size, 2*batch_size, ..., max multiple of batch_size <= num_samples]
        for offset in range(0, num_samples, batch_size):
            # Get the samples you'll use in this batch
            batch_samples = samples[offset:offset+batch_size]

            # Initialise X_train and y_train arrays for this batch
            X_train = []
            y_train = []

            # For each example
            for batch_sample in batch_samples:
                # Load image (X) and label (y)
                img_name = batch_sample[0]
                label = batch_sample[1]
                one_hot = np.zeros(100)
                one_hot[label] = 1
                img =  imread(img_name)

                # apply any kind of preprocessing
                img = resize(img,(224,224))
                # Add example to arrays
                X_train.append(img)
                y_train.append(one_hot)

            # Make sure they're numpy arrays (as opposed to lists)
            X_train = np.array(X_train)
            y_train = np.array(y_train)

            # The generator-y part: yield the next training batch
            yield X_train, y_train


def convert(file_list):
    class_string = os.listdir('pics/train')
    #class_string.remove('.DS_Store')
    class_string.sort()
    class_dict = dict([(string, string_id) for string_id, string in enumerate(class_string)])

    final_list = []
    for file in file_list:
        print(file)
        label = file.split('/')[-2]
        label = class_dict[label]
        final_list.append([file, label])
    return final_list



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
  outer_list = []
  file_list=[]
  for (path,dir,filenames) in os.walk('pics'):
      if not len(filenames)==0:
          for file in filenames:

              file_list.append(os.path.join(path,file))
  correct_file_list = convert(file_list)
  #print(correct_file_list)
  with open("output.txt", "w") as txt_file:
      for line in file_list:
          txt_file.write("".join(line) + "\n")


  datagen_train = ImageDataGenerator(rescale=1/255)
  datagen_test =ImageDataGenerator(rescale=1/255)

  generator_train = datagen_train.flow_from_directory(directory='pics/train',
                                                      target_size = input_shape,
                                                      batch_size=batch_size,
                                                      shuffle = True)

  generator_test = datagen_train.flow_from_directory(directory='pics/test',
                                                     target_size=input_shape,
                                                     batch_size=batch_size,
                                                     shuffle=False)

  steps_test = generator_test.n / batch_size #number of validation steps
  if first_time:
      optimizer = Adam()
  else:
      optimizer = Adam(lr=10**-5)
  loss = 'categorical_crossentropy'
  metrics = ['categorical_accuracy']
  checkpoint = ModelCheckpoint(filename, monitor='loss', verbose=0, save_best_only=True, mode='min')
  callbacks = [checkpoint]

  new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
  new_model.fit_generator(generator=generator(correct_file_list),
                          epochs=epochs,
                          steps_per_epoch = 782,
                          validation_data=generator_test,
                          validation_steps=steps_test,
                          callbacks=callbacks)

