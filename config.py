
class config():
	env_name = "Reacher-v1"
	record = True
	r_seed = 15
	seed_str = 'r_seed=' + str(r_seed)
	# output config
	output_path = "results/{}-{}/".format(env_name, seed_str)
	model_output = output_path + "model.weights/"
	log_path = output_path + "log.txt"
	plot_output = output_path + "scores.png"
	record_path = output_path + "monitor/"
	summary_freq = 1
	
	# model and training config
	num_episodes_test = 50
	max_ep_len = 1000		# maximum episode length
	eval_freq = 5000
	saving_freq = 5000
	record_freq = 5000
	log_freq = 50
	
	
	# hyper parameters
	nsteps_train = 100000
	learning_start = 5000
	learning_freq = 5			# learn after performing every 5 steps
	buffer_size = 1000000		# replay buffer capacity (of transitions)
	weight_decay = 0.01			# for l2 regularizer
	batch_size = 64  # number of steps used to compute each policy update
	actor_lr = 1e-4
	critic_lr = 1e-3
	gamma = 0.99  				# the discount factor
	tau = 0.001  				# soft target update rate

	
