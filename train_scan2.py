from random import shuffle
import os
import tensorflow as tf
import keras.backend as K
import copy
import numpy as np
import math

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
import keras
from keras.losses import categorical_crossentropy, kullback_leibler_divergence
from keras.layers import Dense, Flatten, Input, Conv2D, BatchNormalization, Activation, Conv2DTranspose, Multiply, Dropout
from keras.models import Model, Sequential, load_model
from keras.activations import relu, sigmoid
from keras.optimizers import Adam, RMSprop
from keras.datasets import cifar100

# Imports regarding custom image generator
from skimage.io import imread
from skimage.transform import rescale, resize


def add_module(layer, classifier_name, feature_map_shape=(7, 7, 512)):
    """Attention Module"""
    layer_width=layer.shape[1]
    division_factor = int(layer_width)/int(feature_map_shape[0])
    no_of_twos = int(math.log2(division_factor))
    print(no_of_twos)
    strides_list=[2, 2, 2]
    padding = 'valid'
    if no_of_twos==1:
        strides_list=[1, 2, 1]
        padding = 'same'
    if no_of_twos==2:
        strides_list=[1, 2, 2]
    if no_of_twos==3:
        strides_list=[2, 2, 2]
    if no_of_twos==4:
        strides_list=[2, 4, 2]
    print(strides_list)
    x = Conv2D(filters=int(layer.shape[-1])//2, kernel_size=(2, 2), strides=2)(layer) #Actual kernel size unknown
    x = BatchNormalization()(x)
    x = Activation(relu)(x)
    x = Conv2DTranspose(filters=int(layer.shape[-1]), kernel_size=(2, 2), strides=(2, 2))(x) #Actual kernel size unknown
    x = BatchNormalization()(x)
    x = Activation(sigmoid)(x)
    x.set_shape(layer.shape)

    x = Multiply()([x, layer]) #is this the right dot product? (yes, pretty shure)

    """Bottleneck"""
    """Antal filter går från 64 eller 128 till 512, var sker övergången?"""
    """Dimensionen går från 112 till 14, alltså division med 8, eller 2^3"""
    print(x.shape)
    x = Conv2D(filters=512, kernel_size=(1, 1), strides=strides_list[0])(x)
    x = BatchNormalization()(x)
    x = Activation(relu)(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=strides_list[1],padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation(relu)(x)
    x = Conv2D(filters=512, kernel_size=(1, 1), strides=strides_list[2])(x)
    x = BatchNormalization()(x)
    x = Activation(relu)(x)
    x = Flatten()(x)
    pred_layer = Dense(100, activation='softmax', name=classifier_name)(x)
    return pred_layer

def create_scan_net(model, split_layer_names):
    model_input = Input(shape=(224,224,3), dtype='float32', name='main_input')
    model.layers[1].trainable = False
    X = model.layers[1](model_input)
    pred_outputs = [] # holding all output layers of the scan_net
    i = 1 # used in classifier names
    for j, layer in enumerate(model.layers):
        if j > 1:
            X = layer(X)
            layer.trainable = False
            if layer.name in split_layer_names:
                classifier_name = 'classifier_' + str(i)
                print(classifier_name)
                pred_layer = add_module(X, classifier_name)
                # pred_layer = Flatten()(X)
                # pred_layer = Dense(1000, activation='softmax', name=classifier_name)(pred_layer)
                pred_outputs.append(pred_layer)
                i += 1
    #model.get_layer(name='flatten_1').name='flatten_1_original'
    model.get_layer(name='flatten_1').name='flatten_1_original'

    pred_outputs.append(X)
    scan_model = Model(inputs=model_input, outputs=pred_outputs)
    scan_model.summary()
    #print(len(scan_model.layers))
    return scan_model


def create_vgg_net(model):
    model_inp = model.layers[0].layers[0].input
    X = model.layers[0].layers[1](model_inp)
    i = 1 # used in classifier names
    for j, layer in enumerate(model.layers[0].layers):
        if j > 1:
            X = layer(X)

    X = model.layers[1](X)
    X = model.layers[2](X)
    X = model.layers[3](X)
    X = model.layers[4](X)
    vgg_model = Model(inputs=model_inp, outputs=X)
    vgg_model.summary()
    return vgg_model

def custom_loss(q_c, F_c=0.0, F_i=0.0, alpha=0.5, beta=0.2): # beta corresponds to lambda in the paper
    def loss(y_true, y_pred):
        KLD = kullback_leibler_divergence(q_c, y_pred)
        cross_entropy = categorical_crossentropy(y_true, y_pred)
        F_diff_squared = K.sum(K.square(F_i - F_c))
        loss_value = (1-alpha) * cross_entropy + alpha * KLD + beta * F_diff_squared
        return loss_value
        print("loss is in use")
    return loss

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
            batch_outputs = {
            'classifier_1' : y_train,
            'classifier_2' : y_train,
            'classifier_3' : y_train,
            'dense_2' : y_train
        }

            # The generator-y part: yield the next training batch
            yield X_train, batch_outputs



def convert(file_list):
    class_string = os.listdir('pics/train')
    #class_string.remove('.DS_Store')
    class_string.sort()
    class_dict = dict([(string, string_id) for string_id, string in enumerate(class_string)])

    final_list = []
    for file in file_list:
       # print(file)
        label = file.split('/')[-2]
        label = class_dict[label]
        final_list.append([file, label])
    return final_list


if __name__ == '__main__':
        split_layer_names = ['block2_pool', 'block3_pool', 'block4_pool']
        #model = VGG16(include_top=True, weights='imagenet') #create pretrained VGG16
        model2 = load_model('checkpoint.hdf5')
        vgg_16 = create_vgg_net(model2)
        vgg_16.trainable = False
        vgg_16.summary()

        scan_net = create_scan_net(vgg_16, split_layer_names)
        scan_net.compile(optimizer= Adam(lr=1e-5),
        #loss = [custom_loss(scan_net.get_layer('dense_2').get_output_at(-1), scan_net.get_layer('flatten_1_original').get_output_at(-1), scan_net.get_layer('flatten_1').get_output_at(-1)),
        #custom_loss(scan_net.get_layer('dense_2').get_output_at(-1), scan_net.get_layer('flatten_1_original').get_output_at(-1), scan_net.get_layer('flatten_2').get_output_at(-1)),
        #custom_loss(scan_net.get_layer('dense_2').get_output_at(-1), scan_net.get_layer('flatten_1_original').get_output_at(-1), scan_net.get_layer('flatten_3').get_output_at(-1)),
        #custom_loss(scan_net.get_layer('dense_2').get_output_at(-1), scan_net.get_layer('flatten_1_original').get_output_at(-1), scan_net.get_layer('flatten_1_original').get_output_at(-1))],
        #metrics = ['accuracy'])
        loss = [custom_loss(scan_net.get_layer('dense_2').get_output_at(-1)),custom_loss(scan_net.get_layer('dense_2').get_output_at(-1)),custom_loss(scan_net.get_layer('dense_2').get_output_at(-1)),custom_loss(scan_net.get_layer('dense_2').get_output_at(-1))],
        #loss = ['categorical_crossentropy','categorical_crossentropy','categorical_crossentropy','categorical_crossentropy'],
	metrics = ['accuracy'])
        input_shape = scan_net.layers[0].output_shape[1:3]


        
        file_list = []
        for (path,dir,filenames) in os.walk('pics'):
            if  not len(filenames)==0:
                for file in filenames:

                    file_list.append(os.path.join(path,file))

        correct_file_list = convert(file_list)

        batch_size = 64
        epochs = 20
        steps_per_epoch = 782


        scan_net.fit_generator(generator=generator(correct_file_list),
                                        epochs = epochs,
                                        steps_per_epoch = steps_per_epoch)
