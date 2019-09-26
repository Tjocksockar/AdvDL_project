import tensorflow as tf 

from keras.applications import ResNet50
from keras.applications import ResNet101
from keras.applications import ResNet152

model = ResNet152(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
model.summary()