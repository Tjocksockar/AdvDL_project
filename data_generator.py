import os
import numpy as np
import math

from random import shuffle
from skimage.io import imread
from skimage.transform import rescale, resize

def generator_scan(samples, batch_size=64,shuffle_data=True):
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            X_train = []
            y_train = []
            for batch_sample in batch_samples:
                img_name = batch_sample[0]
                label = batch_sample[1]
                one_hot = np.zeros(100)
                one_hot[label] = 1
                img =  imread(img_name)
                img = resize(img,(224,224))
                X_train.append(img)
                y_train.append(one_hot)

            X_train = np.array(X_train)
            y_train = np.array(y_train)
            batch_outputs = {
            	'classifier_1' : y_train,
            	'classifier_2' : y_train,
            	'classifier_3' : y_train,
            	'dense_2' : y_train
        	}
            yield X_train, batch_outputs

def generator_backbone(samples, batch_size=64,shuffle_data=True):
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            X_train = []
            y_train = []
            for batch_sample in batch_samples:
                img_name = batch_sample[0]
                label = batch_sample[1]
                one_hot = np.zeros(100)
                one_hot[label] = 1
                img =  imread(img_name)
                img = resize(img,(224,224))
                X_train.append(img)
                y_train.append(one_hot)

            X_train = np.array(X_train)
            y_train = np.array(y_train)
            yield X_train, y_train

def create_generator_input(training_fraction=0.85):
    file_list = []
        for (path,dir,filenames) in os.walk('pics'):
            if  not len(filenames)==0:
                for file in filenames:
                    file_list.append(os.path.join(path,file))

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
    final_list = shuffle(final_list)
    cut_off = int(training_fraction * len(final_list))
    train_list = final_list[0:cut_off]
    val_list = final_list[cut_off:len(final_list)]
    return train_list, val_list

