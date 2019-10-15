import tensorflow as tf 
import keras.backend as K
import copy

from keras.applications import ResNet50
from keras.applications import ResNet101
from keras.applications import ResNet152
from keras.applications import VGG16

from keras.losses import categorical_crossentropy, kullback_leibler_divergence
from keras.layers import Dense, Flatten, Input, Conv2D, BatchNormalization, Activation, Conv2DTranspose, Multiply, multiply
from keras.models import Model
from keras.activations import relu, sigmoid

def add_module(layer, classifier_name):
	"""Attention Module"""
	input_copy = copy.copy(layer)
	print(input_copy.shape)
	x = Conv2D(filters=int(layer.shape[-1])//2, kernel_size=(2, 2), strides=2)(layer) #Actual kernel size unknown
	x = BatchNormalization()(x)
	x = Activation(relu)(x)
	x = Conv2DTranspose(filters=int(layer.shape[-1]), kernel_size=(2, 2), strides=(2, 2))(x) #Actual kernel size unknown
	x = BatchNormalization()(x)
	x = Activation(sigmoid)(x)
	print("before dot")
	x.set_shape(input_copy.shape)
	print(x.shape)
	
	x = Multiply()([x, layer]) #is this the right dot product? (yes, pretty shure)
	
	print("after dot")
	print(x.shape)

	"""Bottleneck"""
	"""Antal filter går från 64 eller 128 till 512, var sker övergången?"""
	"""Dimensionen går från 112 till 14, alltså division med 8, eller 2^3"""
	x = Conv2D(filters=512, kernel_size=(1, 1), strides=1)(x)
	x = BatchNormalization()(x)
	x = Activation(relu)(x)
	x = Conv2D(filters=512, kernel_size=(3, 3), strides=1)(x)
	x = BatchNormalization()(x)
	x = Activation(relu)(x)
	x = Conv2D(filters=512, kernel_size=(1, 1), strides=1)(x)
	x = BatchNormalization()(x)
	x = Activation(relu)(x)
	x = Flatten()(x)
	pred_layer = Dense(1000, activation='softmax', name=classifier_name)(x)
	return pred_layer

def create_scan_net(model, split_layer_names): 
	model_input = Input(shape=(224,224,None), dtype='float32', name='main_input')
	X = model.layers[1](model_input)
	pred_outputs = [] # holding all output layers of the scan_net
	i = 1 # used in classifier names
	for j, layer in enumerate(model.layers):
		if j > 1: 
			X = layer(X)
			if layer.name in split_layer_names:
				classifier_name = 'classifier_' + str(i)
				print(classifier_name)
				pred_layer = add_module(X, classifier_name)
				# pred_layer = Flatten()(X)
				# pred_layer = Dense(1000, activation='softmax', name=classifier_name)(pred_layer)
				pred_outputs.append(pred_layer)
				i += 1
	pred_outputs.append(X)
	scan_model = Model(inputs=model_input, outputs=pred_outputs)
	scan_model.summary()
	print(len(scan_model.layers))
	return scan_model

def custom_loss(q_c, F_c=0.0, F_i=0.0, alpha=0.5, beta=0.2): # beta corresponds to lambda in the paper
	def loss(y_true, y_pred): 
		KLD = kullback_leibler_divergence(q_c, y_pred)
		cross_entropy = categorical_crossentropy(y_true, y_pred)
		F_diff_squared = K.sum(K.square(F_i - F_c))
		loss_value = (1-alpha) * cross_entropy + alpha * KLD + beta * F_diff_squared
		return loss_value
	return loss 

if __name__ == '__main__': 
	split_layer_names = ['block2_pool', 'block3_pool']
	model = VGG16(include_top=True, weights='imagenet') #create pretrained VGG16

	scan_net = create_scan_net(model, split_layer_names)
	scan_net.compile(optimizer='Adam', loss = custom_loss(scan_net.get_layer('predictions').get_output_at(-1)), metrics = ['accuracy'])
