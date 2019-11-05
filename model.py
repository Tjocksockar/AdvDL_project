import tensorflow as tf
import keras.backend as K
import copy
import numpy as np
import math

from keras.applications import VGG16, ResNet50, ResNet101, ResNet152
from keras.losses import categorical_crossentropy, kullback_leibler_divergence
from keras.layers import Dense, Flatten, Input, Conv2D, BatchNormalization, Activation, Conv2DTranspose, Multiply, Dropout
from keras.models import Model, Sequential, load_model
from keras.activations import relu, sigmoid
from keras.optimizers import Adam, RMSprop

def add_module(layer, classifier_name, feature_map_shape=(7, 7, 512), classes=100):
	"""Attention Module"""
	layer_width=layer.shape[1]
	no_of_filters=feature_map_shape[2]
	print(layer.shape)
	division_factor = int(layer_width)/int(feature_map_shape[0])
	no_of_twos = int(math.log2(division_factor))
	print(no_of_twos)
	padding = 'valid'
	strides_list=[2, 2, 2]
	if no_of_twos==1:
		padding = 'same'
		strides_list=[1, 2, 1]
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
	x = Conv2D(filters=no_of_filters, kernel_size=(1, 1), strides=strides_list[0])(x)
	x = BatchNormalization()(x)
	x = Activation(relu)(x)
	x = Conv2D(filters=no_of_filters, kernel_size=(3, 3), strides=strides_list[1], padding=padding)(x)
	x = BatchNormalization()(x)
	x = Activation(relu)(x)
	x = Conv2D(filters=no_of_filters, kernel_size=(1, 1), strides=strides_list[2])(x)
	x = BatchNormalization()(x)
	x = Activation(relu)(x)
	x = Flatten()(x)
	pred_layer = Dense(classes, activation='softmax', name=classifier_name)(x)
	return pred_layer

def create_scan_net(model, split_layer_names, feature_map_shape=(7, 7, 512)):
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
    return loss

def loss(y_true, y_pred): # needed for loading trained SCAN model. Known issue in Keras
    loss_value = cross_entropy = categorical_crossentropy(y_true, y_pred)
    return loss_value

def build_resnet_model(classes, version=50, input_shape=(224,224,3)): 
  if version==50:
    model = ResNet50(include_top=False, weights='imagenet',input_shape=input_shape)
  if version==101:
    model = ResNet101(include_top=False, weights='imagenet',input_shape=input_shape)
  if version==152:
    model = ResNet152(include_top=False, weights='imagenet',input_shape=input_shape)
  print('Using ResNet'+str(version))
  transfer_layer = model.layers[-1]
  #conv_model = Model(inputs=model.input, outputs=transfer_layer.output)
  new_model = Sequential()
  new_model.add(model)
  new_model.add(Flatten())
  new_model.add(Dense(1024,activation='relu'))
  new_model.add(Dropout(0.25))
  new_model.add(Dense(classes ,activation='softmax'))
  model.trainable = False
  return input_shape, new_model

def build_resnet_model2(classes, version=50, input_shape=(224,224,3)): 
  if version==50:
    model = ResNet50(include_top=False, weights='imagenet',input_shape=input_shape)
  if version==101:
    model = ResNet101(include_top=False, weights='imagenet',input_shape=input_shape)
  if version==152:
    model = ResNet152(include_top=False, weights='imagenet',input_shape=input_shape)
  model.trainable = False
  model_input = model.input
  X = model.layers[-1]
  #conv_model = Model(inputs=model.input, outputs=transfer_layer.output)
  X = Flatten()(X.output)
  X = Dense(1024,activation='relu')(X)
  X = Dropout(0.25)(X)
  X = Dense(classes ,activation='softmax')(X)
  new_model = Model(inputs=model_input, outputs=X)
  return input_shape, new_model

"""def create_scan_net_resnet(model, split_layer_names=['add_3', 'add_6', 'add_9'], feature_map_shape=(7, 7, 2048)): 
  pred_outputs = []
  for i, layer_name in enumerate(split_layer_names): 
    layer = conv_model.get_layer(layer_name).output
    output_name = 'classifier_' + str(i+1)
    print(output_name)
    pred_layer = add_module(layer, output_name)
    pred_outputs.append(pred_layer)
  pred_outputs.append(model.layers[-1].output)
  scan_net = Model(inputs=model.input, outputs=pred_outputs)
  return scan_net"""

def create_scan_net_resnet(model, split_layer_names): 
  pred_outputs = []
  for i, layer_name in enumerate(split_layer_names): 
    layer = model.get_layer(layer_name).output
    output_name = 'classifier_' + str(i+1)
    print(output_name)
    pred_layer = add_module(layer, output_name)
    pred_outputs.append(pred_layer)
  pred_outputs.append(model.layers[-1].output)
  scan_net = Model(inputs=model.input, outputs=pred_outputs)
  return scan_net

# def create_scan_net_resnet(model, split_layer_names, feature_map_shape=(7, 7, 2048)):
#     model.summary()
#     model_input = model.layers[0].layers[0].output#Input(shape=(224,224,3), dtype='float32', name='main_input')
#     #model.layers[0].trainable = False
#     X = model.layers[0].layers[1](model_input)
#     pred_outputs = [] # holding all output layers of the scan_net
#     i = 1 # used in classifier names
#     for j, layer in enumerate(model.layers[0].layers):
#         if j > 1:
#             print(j)
#             X = layer(X)
#             layer.trainable = False
#             if layer.name in split_layer_names:
#                 classifier_name = 'classifier_' + str(i)
#                 print(classifier_name)
#                 pred_layer = add_module(X, classifier_name, feature_map_shape)
#                 # pred_layer = Flatten()(X)
#                 # pred_layer = Dense(1000, activation='softmax', name=classifier_name)(pred_layer)
#                 pred_outputs.append(pred_layer)
#                 i += 1
#     final_output = model.layers[-1].output
#     pred_outputs.append(final_output)
#     #model.get_layer(name='flatten_1').name='flatten_1_original'
#     model.get_layer(name='flatten_1').name='flatten_1_original'

#     pred_outputs.append(X)
#     scan_model = Model(inputs=model_input, outputs=pred_outputs)
#     scan_model.summary()
#     #print(len(scan_model.layers))
#     return scan_model