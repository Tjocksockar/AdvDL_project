from genetic_algorithm import *
from data_generator import *
from random import shuffle

scan_net = load_model('scan.hdf5')

train_list, val_list = create_generator_input()

### Find threshold
predictions = scan_net.predict_generator(generator=generator_predict(val_list), steps = 1)
threshold = finding_threshold(predictions)
print('='*50)
print('This is the final threshold')
print(threshold)                       
print('='*50)

### Check how many times each classifier is used based on the found threshold
train_list, val_list = create_generator_input()
shuffle(val_list)
n_samples = 1000
small_val_list = val_list[0:n_samples]
predictions = scan_net.predict_generator(generator=generator_predict_with_labels(small_val_list), steps = n_samples)
used_times = get_classifier_used(threshold, predictions)
print('the classifiers used')
print(used_times)
print('='*50)

# Get acceleration ratio and accuracy
acc_ratio, accuracy = get_accuracy_vs_acceleration(threshold, small_val_list, predictions)
print('the acceleration ratio')
print(acc_ratio)
print('the accuracy')
print(accuracy)
print('='*50)