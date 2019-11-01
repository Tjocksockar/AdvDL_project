from random import randint
from data_generator import *

def finding_threshold(predictions, N=30, generations=50, population_size=3): 
	parents = [] # initializing random population
	for i in range(population_size):
		parent = []
		for j in range(N):  
			parent.append(randint(0,1))
		parents.append(parent)

	# starting evolution
	for i in range(generations): 
		for vector in parents: 
			print(vector)
		print('='*50)
		thresholds = [] # decode to find thresholds from genes
		for j in range(population_size): 
			thresholds.append(decode_threshold(parents[j]))

		ranked_list = [] # calculate fitness and rank
		for k in range(population_size): 
			ranked_list.append(fitness(thresholds[k], predictions))

		# fittest parents
		best_val = max(ranked_list)
		for index, value in enumerate(ranked_list): 
			if value == best_val: 
				best_idx = index
				ranked_list[index] = -1000
				break
		best_1 = parents[best_idx]
		if i < generations-1: 
			best_val = max(ranked_list)
			for index, value in enumerate(ranked_list): 
				if value == best_val: 
					best_idx = index
					break
			best_2 = parents[best_idx]

			# creating children 
			children = []
			for l in range(population_size): 
				if k > 0: 
					best_1 = mutate(best_1)
					best_2 = mutate(best_2)
				idx_1 = randint(0, N-1)
				idx_2 = randint(0, N-1)
				if idx_1 > idx_2: 
					idx_temp = idx_1
					idx_1 = idx_2 
					idx_2 = idx_temp
				child = best_1[0:idx_1] + best_2[idx_1:idx_2] + best_1[idx_2:N]
				children.append(child)
			parents = children
		else: 
			final_threshold = decode_threshold(best_1)
			return final_threshold

def mutate(gene): 
	idx = randint(0, len(gene)-1)
	if gene[idx] == 0: 
		gene[idx] = 1
	else: 
		gene[idx] = 0
	return gene

def decode_threshold(gene): 
	summation = 0
	for bit in gene: 
		summation += bit
	threshold = 1 - (0.3/len(gene))*summation
	return threshold

def fitness(threshold, predictions, accuracies=[0.3706, 0.4051, 0.4145, 0.2450], baseline=0.2330, beta=0.5): 
	time_consumption = [0.25, 0.5, 0.75, 1.0]
	acceleration_den = 0
	tot_accuracy_diff = 0
	for i in range(predictions[0].shape[0]): 
		for j in range(len(predictions)): 
			softmax = max(predictions[j][i, :])
			if softmax >= threshold or j==3: 
				acceleration_den += time_consumption[j]
				this_acc = accuracies[j]
				tot_accuracy_diff = tot_accuracy_diff + this_acc - baseline
				break
	tot_accuracy_diff = tot_accuracy_diff/predictions[0].shape[0]
	acceleration_ratio = 1 - (acceleration_den / predictions[0].shape[0])
	score = acceleration_ratio + beta * tot_accuracy_diff
	return score

def get_classifier_used(threshold, predictions): 
	statistics = [] # idx 0 : shallowest classifier and so on ...
	for k in range(len(predictions)): 
		statistics.append(0)
	n_predictions = predictions[0].shape[0]
	for i in range(n_predictions): 
		for j in range(len(predictions)): 
			softmax = max(predictions[j][i, :])
			if softmax >= threshold or j==3: 
				statistics[j] += 1
				break
	for m in range(len(statistics)): 
		statistics[m] = statistics[m] / n_predictions
	return statistics

def get_accuracy_vs_acceleration(threshold, val_list, predictions): 
	class_string = os.listdir('pics/train')
	class_string.sort()
	print(class_string)
	if '.DS_Store' in  class_string:
		class_string.remove('.DS_Store')
	class_dict = dict([(string, string_id) for string_id, string in enumerate(class_string)])
	print(len(class_dict))

	time_consumption = [0.25, 0.5, 0.75, 1.0]
	acceleration_den = 0
	correct_preds = 0
	for i in range(predictions[0].shape[0]): 
		for j in range(len(predictions)): 
			softmax = max(predictions[j][i, :])
			if softmax >= threshold or j==3: 
				acceleration_den += time_consumption[j]
				for k, this_softmax in enumerate(predictions[j][i, :]): 
					if this_softmax == softmax: 
						pred_class = k
						true_class = val_list[i].split('/')[-2]
						if class_dict[true_class] == pred_class: 
							correct_preds += 1
				break
	acceleration_ratio = 1 - (acceleration_den / predictions[0].shape[0])
	accuracy = correct_preds / predictions[0].shape[0]
	return acceleration_ratio, accuracy

