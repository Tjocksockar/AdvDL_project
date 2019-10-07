import tensorflow as tf 

from keras.applications import ResNet50
from keras.applications import ResNet101
from keras.applications import ResNet152
from keras.applications import VGG16

from keras.layers import Dense, Flatten

class Scan_net(): 
	def __init__(self, model, split_layer_names): 
		model_input = model.inputs[0]
		print(model_input.name)
		layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
		for i, layer_name in enumerate(split_layer_names):
			classifier_name = 'classifier_' + str(i+1)
			print(classifier_name)
			split_layer = layer_dict[layer_name].output[-1]
			split_layer = Flatten()(split_layer)
			pred_layer = Dense(100, activation='softmax', name=classifier_name)(split_layer)
		model.summary()

if __name__ == '__main__': 
	split_layer_names = ['block2_pool', 'block3_pool']
	model = VGG16(include_top=True, weights='imagenet') #create pretrained VGG16
	model.summary()
	scan_net = Scan_net(model, split_layer_names)