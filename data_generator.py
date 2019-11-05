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

def create_generator_input(training_fraction=0.85, dataset_filename='pics'):
    file_list = []
    for (path,dir,filenames) in os.walk(dataset_filename):
        if  not len(filenames)==0 or '.DS_Store' not in filenames[0]:
            for file in filenames:
                file_list.append(os.path.join(path,file))
    if '.DS_Store' in  file_list:
        file_list.remove('.DS_Store')
    if dataset_filename+'/.DS_Store' in  file_list: 
       file_list.remove(dataset_filename+'/.DS_Store')
    if dataset_filename+'/test/.DS_Store' in  file_list:
       file_list.remove(dataset_filename+'/test/.DS_Store')
    if dataset_filename+'/train/.DS_Store' in  file_list:
       file_list.remove(dataset_filename+'/train/.DS_Store')
    class_string = os.listdir(dataset_filename+'/train')
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
        print(label)
        label = class_dict[label]
        final_list.append([file, label])
    shuffle(final_list)
    cut_off = int(training_fraction * len(final_list))
    train_list = final_list[0:cut_off]
    val_list = final_list[cut_off:len(final_list)]
    return train_list, val_list

def generator_scan_imagenet(samples, batch_size=64,shuffle_data=True):
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
                one_hot = np.zeros(200)
                one_hot[label] = 1
                img =  imread(img_name)
                img = resize(img,(224,224,3))
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

def generator_backbone_imagenet(samples, batch_size=64,shuffle_data=True):
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
                one_hot = np.zeros(200)
                one_hot[label] = 1
                img =  imread(img_name)
                img = resize(img,(224,224,3))
                X_train.append(img)
                y_train.append(one_hot)

            X_train = np.array(X_train)
            y_train = np.array(y_train)
            yield X_train, y_train

def create_generator_input_imagenet(training_fraction=0.85, dataset_filename='tiny-imagenet-200'):
    file_list = []
    for (path, dir, filenames) in os.walk(dataset_filename):
        #print(filenames)
        #print(path)
        if not len(filenames)==0:
            for file in filenames:
                if '.DS_Store' in file:
                    os.remove('/Users/Gustav/Documents/ADL/reimplementation_SCAN/AdvDL_project/'+path+'/'+file)
                else:
                    file_list.append(os.path.join(path,file))
    if '.DS_Store' in  file_list:
        file_list.remove('.DS_Store')
    if dataset_filename+'/.DS_Store' in  file_list: 
       file_list.remove(dataset_filename+'/.DS_Store')
    if dataset_filename+'/train/.DS_Store' in  file_list:
       file_list.remove(dataset_filename+'/train/.DS_Store')

    #print(file_list)
    class_string = os.listdir(dataset_filename+'/train')
    class_string.sort()
    #print(class_string)
    if '.DS_Store' in  class_string:
        class_string.remove('.DS_Store')
    class_dict = dict([(string, string_id) for string_id, string in enumerate(class_string)])
    print(len(class_dict))
    print(class_dict)

    final_list = []
    for file in file_list:
        #print(file)
        label = file.split('/')[-3]
        label = class_dict[label]
        final_list.append([file, label])
    shuffle(final_list)
    cut_off = int(training_fraction * len(final_list))
    train_list = final_list[0:cut_off]
    val_list = final_list[cut_off:len(final_list)]
    return train_list, val_list


