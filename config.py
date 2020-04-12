import tensorflow as tf

class config():
	env_name = "Reacher-v2"
	record = True
	r_seed = 15
	seed_str = 'r_seed=' + str(r_seed)
	# output config
	output_path = "results/{}-{}/".format(env_name, seed_str)
	model_output = output_path + "model.weights/"
	log_path = output_path + "log.txt"
	plot_output = output_path + "scores.png"
	record_path = output_path + "monitor/"
	record_freq = 5
	summary_freq = 1
	
	# model and training config
	batch_size = 64			# number of steps used to compute each policy update
	max_ep_len = 1000		# maximum episode length
	actor_learning_rate = 1e-4
	critic_learning_rate = 1e-3
	gamma = 0.99			# the discount factor
	tau = 0.001				# soft target update rate
	buffer_size = 1000000	# replay buffer capacity (of transitions)
	weight_decay = 0.01     # for l2 regularizer
	
