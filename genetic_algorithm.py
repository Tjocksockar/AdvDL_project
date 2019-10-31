from random import randint

def finding_threshold(N=10, generations=50, population_size=3): 
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
			ranked_list.append(fitness(thresholds[k]))

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

def fitness(threshold): 
	return threshold

if __name__ == '__main__': 
	threshold = finding_threshold()
	print(threshold)