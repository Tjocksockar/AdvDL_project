import model
import data_generator

split_layer_names = ['block2_pool', 'block3_pool', 'block4_pool']
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
metrics = ['accuracy'])
input_shape = scan_net.layers[0].output_shape[1:3]

batch_size = 64
epochs = 200

train_list, val_list = create_generator_input()
steps_per_epoch = int(len(train_list)/batch_size)
val_steps_per_epoch = int(len(val_list)/batch_size)

scan_net.fit_generator(generator=generator_scan(train_list),
                        epochs = epochs,
                        steps_per_epoch = steps_per_epoch, 
                        validation_data = generator_scan(val_list), 
                        validation_steps = val_steps_per_epoch)
              