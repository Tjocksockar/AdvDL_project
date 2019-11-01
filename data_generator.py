import os
import numpy as np
import math
import random 

from random import shuffle
from skimage.io import imread
from skimage.transform import rescale, resize
#from keras.preprocessing.image import apply_transform

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
                apply_aug = random.randint(0,1)
                if apply_aug: 
                    img = np.flipud(img)
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
                apply_aug = random.randint(0,1)
                if apply_aug: 
                    img = np.flipud(img)
                X_train.append(img)
                y_train.append(one_hot)

            X_train = np.array(X_train)
            y_train = np.array(y_train)
            yield X_train, y_train

def create_generator_input():
    file_list = []
    for (path,dir,filenames) in os.walk('pics'):
        if  not len(filenames)==0 or '.DS_Store' not in filenames[0]:
            for file in filenames:
                file_list.append(os.path.join(path,file))
    if '.DS_Store' in  file_list:
        file_list.remove('.DS_Store')
    if 'pics/.DS_Store' in  file_list: 
       file_list.remove('pics/.DS_Store')
    if 'pics/test/.DS_Store' in  file_list:
       file_list.remove('pics/test/.DS_Store')
    if 'pics/train/.DS_Store' in  file_list:
       file_list.remove('pics/train/.DS_Store')
    class_string = os.listdir('pics/train')
    class_string.sort()
    print(class_string)
    if '.DS_Store' in  class_string:
        class_string.remove('.DS_Store')
    class_dict = dict([(string, string_id) for string_id, string in enumerate(class_string)])
    print(len(class_dict))

    final_list = []
    for file in file_list:
        #print(file)
        label = file.split('/')[-2]
        label = class_dict[label]
        final_list.append([file, label])
    train_list = []
    val_list = []
    for element in final_list: 
        if 'train' in element[0]: 
            train_list.append(element)
        else: 
            val_list.append(element)    
    return train_list, val_list

def generator_predict(samples, batch_size=64,shuffle_data=True):
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            X_train = []
            for batch_sample in batch_samples:
                img_name = batch_sample[0]
                img =  imread(img_name)
                img = resize(img,(224,224))
                X_train.append(img)

            X_train = np.array(X_train)
            yield X_train

def generator_predict_with_labels(samples): 
    while True: 
        x_batch = []
        for batch_sample in samples: 
            img_name = batch_sample[0]
            img =  imread(img_name)
            img = resize(img,(224,224))
            x_batch.append(img)
        X_train = np.array(x_batch)
        yield X_train