import os
import tensorflow as tf
import keras.backend as K
import copy
import numpy as np

# from keras.applications import ResNet50
# from keras.applications import ResNet101
# from keras.applications import ResNet152
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

# Imports regarding custom image generator
from skimage.io import imread
from skimage.transform import rescale, resize


def add_module(layer, classifier_name, feature_map_shape=(7, 7, 512)):
    """Attention Module"""
    layer_width=layer.shape[1]
    division_factor = int(layer_width)/int(feature_map_shape[0])
    no_of_twos = int(math.log2(division_factor))
    print(no_of_twos)
    strides_list=[2, 2, 2]
    if no_of_twos==1:
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
    x = Conv2D(filters=512, kernel_size=(1, 1), strides=strides_list[0])(x)
    x = BatchNormalization()(x)
    x = Activation(relu)(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=strides_list[1])(x)
    x = BatchNormalization()(x)
    x = Activation(relu)(x)
    x = Conv2D(filters=512, kernel_size=(1, 1), strides=strides_list[2])(x)
    x = BatchNormalization()(x)
    x = Activation(relu)(x)
    x = Flatten()(x)
    pred_layer = Dense(200, activation='softmax', name=classifier_name)(x)
    return pred_layer

def create_scan_net(model, split_layer_names):
    model_input = Input(shape=(224,224,3), dtype='float32', name='main_input')
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
    #model.get_layer(name='flatten_1').name='flatten_1_original'
    model.get_layer(name='flatten_1').name='flatten_1_original'

    pred_outputs.append(X)
    scan_model = Model(inputs=model_input, outputs=pred_outputs)
    scan_model.summary()
    print(len(scan_model.layers))
    return scan_model


def create_vgg_net(model):
    #model_input = Input(shape=(32,32,None), dtype='float32', name='main_input')
    #awt = model.layers[4].output
    model_inp = model.layers[0].layers[0].input
    print(model_inp)
    X = model.layers[0].layers[1](model_inp)
    i = 1 # used in classifier names
    for j, layer in enumerate(model.layers[0].layers):
        if j > 1:
            X = layer(X)

    #print(model.layers[4].get_output_at(0))
    #print(model.layers[4].get_output_at(1))
    X = model.layers[1](X)
    X = model.layers[2](X)
    X = model.layers[3](X)
    X = model.layers[4](X)
    vgg_model = Model(inputs=model_inp, outputs=X)
    #vgg_model = Model(inputs=model_inp, outputs=awt)
    vgg_model.summary()
    return vgg_model

def custom_loss(q_c, F_c=0.0, F_i=0.0, alpha=0.5, beta=0.2): # beta corresponds to lambda in the paper
    def loss(y_true, y_pred):
        KLD = kullback_leibler_divergence(q_c, y_pred)
        cross_entropy = categorical_crossentropy(y_true, y_pred)
        F_diff_squared = K.sum(K.square(F_i - F_c))
        loss_value = (1-alpha) * cross_entropy + alpha * KLD + beta * F_diff_squared
        return loss_value
        print("loss is in use")
    return loss




def custom_img_generator(filepaths, batch_size=64):
    class_string = os.listdir('pics/train')
    print(class_string)
    class_string.remove('.DS_Store')
    class_string.sort()
    print(class_string)
    #print(class_string)
    #print(class_string)
    class_dict = dict([(string, string_id) for string_id, string in enumerate(class_string)])
    #print(class_dict)



    while True:
        batch_paths = np.random.choice(a=filepaths, size=batch_size)
        batch_input = []
        batch_output = []
        for path in batch_paths:
            img = imread(path)
            #print(img.shape)
            label = path.split('/')[-2]
            #print(label)
            label = class_dict[label]
            #img = rescale(img, 1./255, anti_aliasing=False)
            #print(img.shape)
            img = resize(img, (224, 224), anti_aliasing=False)

            batch_input.append(img)
            batch_output.append(label)
        batch_x = np.array(batch_input)*(1./255)
        batch_y = np.array(batch_output)
        batch_outputs = {
            'classifier_1' : batch_y,
            'classifier_2' : batch_y,
            'classifier_3' : batch_y,
            'dense_2' : batch_y
        }
        yield(batch_x, batch_outputs)


if __name__ == '__main__':
        split_layer_names = ['block2_pool', 'block3_pool', 'block4_pool']
        #model = VGG16(include_top=True, weights='imagenet') #create pretrained VGG16
        model2 = load_model('my_model3.h5')
        vgg_16 = create_vgg_net(model2)

        scan_net = create_scan_net(vgg_16, split_layer_names)
        scan_net.compile(optimizer='Adam',
        # loss = [custom_loss(scan_net.get_layer('dense_2').get_output_at(-1), scan_net.get_layer('flatten_1_original').get_output_at(-1), scan_net.get_layer('flatten_1').get_output_at(-1)),
        # custom_loss(scan_net.get_layer('dense_2').get_output_at(-1), scan_net.get_layer('flatten_1_original').get_output_at(-1), scan_net.get_layer('flatten_2').get_output_at(-1)),
        # custom_loss(scan_net.get_layer('dense_2').get_output_at(-1), scan_net.get_layer('flatten_1_original').get_output_at(-1), scan_net.get_layer('flatten_3').get_output_at(-1)),
        # custom_loss(scan_net.get_layer('dense_2').get_output_at(-1), scan_net.get_layer('flatten_1_original').get_output_at(-1), scan_net.get_layer('flatten_1_original').get_output_at(-1))],
        # metrics = ['accuracy'])
        loss = [custom_loss(scan_net.get_layer('dense_2').get_output_at(-1)),custom_loss(scan_net.get_layer('dense_2').get_output_at(-1)),custom_loss(scan_net.get_layer('dense_2').get_output_at(-1)),custom_loss(scan_net.get_layer('dense_2').get_output_at(-1))],
        metrics = ['accuracy'])
        input_shape = scan_net.layers[0].output_shape[1:3]

        #print(input_shape)
        datagen_train = ImageDataGenerator(rescale=1./255)
        #
        datagen_test =ImageDataGenerator(rescale=1./255)

        batch_size = 20

        # generator_train = datagen_train.flow_from_directory(directory='pics/train',
        #                                                     target_size = input_shape,
        #                                                     batch_size=batch_size,
        #                                                     shuffle = True)
        # #
        # generator_test = datagen_train.flow_from_directory(directory='pics/test',
        #                                                    target_size=input_shape,
        #                                                    batch_size=batch_size,
        #                                                    shuffle=False)

        #steps_test = generator_test.n / batch_size
        file_list = []
        # for (root,dirs,files) in os.walk('pics', topdown=True):
        #     #print (root)
        #     #print (dirs)
        #     #print (files)
        #     for (ind,text) in enumerate(files):
        #         if ind > 1:
        #             #print("this is the text " + text)
        #             file_list.append(root+'/'+text)


        for (path,dir,filenames) in os.walk('pics'):
            if '.DS_Store' not in filenames[0]:
                for file in filenames:

                    file_list.append(os.path.join(path,file))

        #print(hej)
        with open("output.txt", "w") as txt_file:
            for line in file_list:
                txt_file.write("".join(line) + "\n")

        epochs = 20
        steps_per_epoch = 100


        #new_model.fit(x_train_new,y_train,batch_size=batch_size,epochs=epochs,validation_data=(x_test_new,y_test))
        # scan_net.fit_generator(generator=generator_train,
        #                         epochs=epochs,
        #                         steps_per_epoch = steps_per_epoch,
        #                         validation_data=generator_test,
        #                         validation_steps=steps_test)

        scan_net.fit_generator(generator=custom_img_generator(file_list),
                                        epochs=epochs,
                                        steps_per_epoch = steps_per_epoch)

                #scan_net.fit_generator(generator=custom_img_generator())
