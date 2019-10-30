import os
import tensorflow as tf
import keras.backend as K
import copy
import numpy as np
import math

from random import shuffle
from skimage.io import imread
from skimage.transform import rescale, resize


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
