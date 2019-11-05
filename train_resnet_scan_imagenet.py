from model import*
from data_generator import*
from keras.models import load_model
from keras.utils import plot_model

first_time=True
dataset_filename = 'tiny-imagenet-200'
TIN = "TinyImageNet"
CIFAR = "CIFAR"
dataset = TIN
version=50
filename="checkpoint_"+dataset+"ResNet"+str(version)+".hdf5"
input_shape=(224,224)
if dataset==TIN:
	classes = 200
else:
	classes = 100
if version == 50:
	split_layer_names = ['add_3', 'add_6', 'add_9']
if version == 101:
	split_layer_names = ['conv2_block1_add', 'conv2_block3_add', 'conv3_block2_add']
if version == 152:
	split_layer_names = ['conv2_block1_add', 'conv2_block3_add', 'conv3_block2_add']

train_list, val_list = create_generator_input_imagenet(dataset_filename=dataset_filename)
if first_time:
	input_shape, new_model = build_resnet_model(classes=classes, version=version)
	plot_model(new_model.layers[0], to_file='resnet_model.png')
	new_model.layers[0].summary()
	scan_resnet = create_scan_net_resnet(new_model, split_layer_names, feature_map_shape=(7, 7, 2048))
else:
	new_model = load_model(filename)
print(new_model.layers[0].layers[0].output.shape)
batch_size = 64
epochs = 1024


steps_test = len(val_list) // batch_size #number of validation steps
steps_train = len(train_list) // batch_size

optimizer = Adam(lr=10**-5)
loss = [custom_loss(scan_net.get_layer('dense_2').get_output_at(-1), scan_net.get_layer('flatten_1_original').get_output_at(-1), scan_net.get_layer('flatten_1').get_output_at(-1)),
		custom_loss(scan_net.get_layer('dense_2').get_output_at(-1), scan_net.get_layer('flatten_1_original').get_output_at(-1), scan_net.get_layer('flatten_2').get_output_at(-1)),
		custom_loss(scan_net.get_layer('dense_2').get_output_at(-1), scan_net.get_layer('flatten_1_original').get_output_at(-1), scan_net.get_layer('flatten_3').get_output_at(-1)),
		custom_loss(scan_net.get_layer('dense_2').get_output_at(-1), scan_net.get_layer('flatten_1_original').get_output_at(-1), scan_net.get_layer('flatten_1_original').get_output_at(-1))]
metrics = ['accuracy']

checkpoint = ModelCheckpoint(filename, monitor='loss', verbose=0, save_best_only=True, mode='min')
callbacks = [checkpoint]

new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
new_model.fit_generator(generator=generator_scan_imagenet(train_list),
                      epochs=epochs,
                      steps_per_epoch = steps_train,
                      validation_data=generator_scan_imagenet(val_list),
                      validation_steps=steps_test,
                      callbacks=callbacks)


          

