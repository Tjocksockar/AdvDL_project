# import tensorflow as tf
# import numpy as np
# import os
# import keras
# from tensorflow.python.keras.models import Model, Sequential
# from tensorflow.python.keras.layers import Dense, Flatten, Dropout
# from tensorflow.python.keras.applications import VGG16
# from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
# from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.python.keras.optimizers import Adam, RMSprop
import tensorflow as tf
import keras.backend as K
import copy
import scipy
import numpy as np

# from keras.applications import ResNet50
# from keras.applications import ResNet101
# from keras.applications import ResNet152
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
import keras
from keras.losses import categorical_crossentropy, kullback_leibler_divergence
from keras.layers import Dense, Flatten, Input, Conv2D, BatchNormalization, Activation, Conv2DTranspose, Multiply, Dropout
from keras.models import Model, Sequential
from keras.activations import relu, sigmoid
from keras.optimizers import Adam, RMSprop
from keras.datasets import cifar100

# (x_train, y_train), (x_test,y_test) = cifar100.load_data()
# y_train = keras.utils.to_categorical(y_train,100)
# y_test = keras.utils.to_categorical(y_test,100)
model = VGG16(include_top=False, weights='imagenet',input_shape=(224,224,3))
input_shape = model.layers[0].output_shape[1:3]
print(input_shape)
datagen_train = ImageDataGenerator(rescale=1./255)
#
datagen_test =ImageDataGenerator(rescale=1./255)

batch_size = 20

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

# image_paths_train
# image_paths_test

# cls_train = generator_train.classes
#
# cls_test = generator_test.classes
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255


## x_train_new = np.empty(shape=(x_train.shape[0],)+new_shape)
# x_test_new = np.empty(shape=(x_test.shape[0],)+new_shape)
#
# for idx in range(x_train.shape[0]):
#     x_train_new[idx] = scipy.misc.imresize(x_train[idx], new_shape)
#
# for idx in range(x_test.shape[0]):
#     x_test_new[idx] = scipy.misc.imresize(x_test[idx], new_shape)

#x_train_new = [skimage.transform.resize(image, new_shape) for image in x_train]
#x_test_new = [skimage.transform.resize(image, new_shape) for image in x_test]


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
