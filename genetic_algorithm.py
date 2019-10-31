def fitness(acceleration_ratio, accuracy, baseline, beta): 
	return acceleration_ratio + beta * (accuracy - baseline)