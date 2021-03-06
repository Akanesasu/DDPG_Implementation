import torch

class Reacher_config():
	env_name = "Reacher-v1"
	record = True
	r_seed = 1234567
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
	eval_freq = 250000
	saving_freq = 250000
	record_freq = 250000
	log_freq = 200
	
	
	# hyper parameters
	nsteps_train = 5000000
	learning_start = 50000
	learning_freq = 5			# learn after performing every 5 steps
	buffer_size = 1000000		# replay buffer capacity (of transitions)
	weight_decay = 0.01			# for l2 regularizer
	batch_size = 64  # number of steps used to compute each policy update
	actor_lr = 1e-4
	critic_lr = 1e-3
	gamma = 0.99  				# the discount factor
	tau = 0.001  				# soft target update rate
	
	# device
	device = None
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available()
								   else "cpu")

class Pendulum_config(object):
	env_name = "Pendulum-v0"
	record = True
	r_seed = 25
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
	max_ep_len = 1000  # maximum episode length
	eval_freq = 50000
	saving_freq = 50000
	record_freq = 50000
	log_freq = 200
	
	# hyper parameters
	nsteps_train = 500000
	learning_start = 50000
	learning_freq = 5  # learn after performing every 5 steps
	buffer_size = 1000000  # replay buffer capacity (of transitions)
	weight_decay = 0.002  # for l2 regularizer
	batch_size = 64  # number of steps used to compute each policy update
	actor_lr = 1e-4
	critic_lr = 1e-3
	gamma = 0.99  # the discount factor
	tau = 0.001  # soft target update rate
	
	# device
	device = None
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available()
							  else "cpu")
	
config = Reacher_config()