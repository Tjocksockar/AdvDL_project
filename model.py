import tensorflow as tf
import keras.backend as K
import copy
import numpy as np
import math

from keras.applications import VGG16
from keras.losses import categorical_crossentropy, kullback_leibler_divergence
from keras.layers import Dense, Flatten, Input, Conv2D, BatchNormalization, Activation, Conv2DTranspose, Multiply, Dropout
from keras.models import Model, Sequential, load_model
from keras.activations import relu, sigmoid
from keras.optimizers import Adam, RMSprop

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