import tensorflow as tf 

from keras.applications import ResNet50
from keras.applications import ResNet101
from keras.applications import ResNet152
from keras.applications import VGG16

from keras.layers import Dense, Flatten, Input
from keras.models import Model

class Scan_net(): 
	def __init__(self, model, split_layer_names): 
		model_input = Input(shape=(224,224,None), dtype='float32', name='main_input')
		X = model.layers[1](model_input)
		pred_outputs = [] # holding all output layers of the scan_net
		i = 1 # used in classifier names
		for j, layer in enumerate(model.layers):
			if j > 1: 
				X = layer(X)
				if layer.name in split_layer_names:
					classifier_name = 'classifier_' + str(i+1)
					print(classifier_name)
					pred_layer = Flatten()(X)
					pred_layer = Dense(100, activation='softmax', name=classifier_name)(pred_layer)
					pred_outputs.append(pred_layer)
					i += 1
		pred_outputs.append(X)
		scan_model = Model(inputs=model_input, outputs=pred_outputs)
		scan_model.summary()
		print(len(scan_model.layers))

if __name__ == '__main__': 
	split_layer_names = ['block2_pool', 'block3_pool']
	model = VGG16(include_top=True, weights='imagenet') #create pretrained VGG16
	#model.summary()
	#print(len(model.layers))
	scan_net = Scan_net(model, split_layer_names)