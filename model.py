import tensorflow as tf 

from keras.applications import ResNet50
from keras.applications import ResNet101
from keras.applications import ResNet152
from keras.applications import VGG16

from keras.layers import Dense, Flatten
from keras.models import Model

class Scan_net(): 
	def __init__(self, model, split_layer_names): 
		model_input = model.inputs[0]
		#layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
		i = 1
		for layer in model.layers:
			if layer.name in split_layer_names:
				classifier_name = 'classifier_' + str(i+1)
				print(classifier_name)
				split_layer = layer.output[-1]
				split_layer = Flatten()(split_layer)
				pred_layer = Dense(100, activation='softmax', name=classifier_name)(split_layer)
				i += 1
		model.summary()
		print('='*70)
		print(len(model.layers))

class Scan_net2(): 
	def __init__(self, model, split_layer_names): 
		model_input = model.layers[0].input
		print(model_input)
		X = model.layers[0](model_input)
		pred_outputs = [] # holding all output layers of the scan_net
		i = 1 # used in classifier names
		for j, layer in enumerate(model.layers):
			if j != 0: 
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
	scan_net = Scan_net2(model, split_layer_names)