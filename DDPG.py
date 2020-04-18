import os
import gym
import numpy as np
import sys
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
from tensorflow.keras.layers import Dense
from keras.utils import Progbar
from critic import Critic
from actor import Actor
from replay_buffer import ReplayBuffer
from OrnsteinUhlenbeck import OrnsteinUhlenbeckActionNoise
from general_utils import *


class DDPG(object):
	"""
	Abstract Class for implementing Deep Deterministic Policy Gradient
	"""
	
	def __init__(self, env, config, logger=None):
		"""
		Initialize Policy Gradient Class
		Args:
				env: an OpenAI Gym environment
				config: class with hyperparameters
		"""
		# directory for training outputs
		if not os.path.exists(config.output_path):
			os.makedirs(config.output_path)
		
		# store hyperparameters
		self.config = config
		self.logger = logger
		if logger is None:
			self.logger = get_logger(config.log_path)
		self.r_seed = config.r_seed
		self.env = env
		self.env.seed(self.r_seed)
		
		self.observation_dim = self.env.observation_space.shape[0]
		self.action_dim = self.env.action_space.shape[0]
		
		self.lr = config.actor_learning_rate
		self.tau = config.tau
		
		# build model
		self.build()
	

	
	def add_soft_target_update_op(self, a_scope, target_a_scope):
		"""
		Move target weights slowly toward eval weights.
		"""
		mu_vars = tfv1.get_collection(tfv1.GraphKeys.TRAINABLE_VARIABLES,
									  scope=a_scope)
		tar_vars = tfv1.get_collection(tfv1.GraphKeys.TRAINABLE_VARIABLES,
									   scope=target_a_scope)
		
		upd_vars = [tar_vars[i] + self.tau * (mu_vars[i] - tar_vars[i])
					for i in range(len(mu_vars))]
		self.update_target_op = tfv1.group(*[tfv1.assign(tar_var, upd_var)
											 for tar_var, upd_var in zip(tar_vars, upd_vars)])
	
	def add_hard_target_update_op(self, a_scope, target_a_scope):
		"""
		Make target weights the same as eval weights.
		"""
		mu_vars = tfv1.get_collection(tfv1.GraphKeys.TRAINABLE_VARIABLES,
									  scope=a_scope)
		tar_vars = tfv1.get_collection(tfv1.GraphKeys.TRAINABLE_VARIABLES,
									   scope=target_a_scope)
		
		self.hard_update_target_op = tfv1.group(*[tfv1.assign(tar_var, mu_var)
												  for tar_var, mu_var in zip(tar_vars, mu_vars)])
	
	def add_loss_op(self, q):
		"""
		Args:
			q (tf tensor): Q(s_t, μ(s_t))


		Gradient Ascent (formula without minibatch):
				θ = θ + ∇_a(Q(s_t, a)|a=μ(s_t)) * ∇_θ(μ(s_t|θ))
				which is equivalent to (by chain rule of derivative)
				θ = θ + ∇_θ(Q(s_t, μ(s_t|θ)))
				So we use -Q(s_t, μ(s_t)) as loss.
		"""
		self.loss = -q
	
	def add_optimizer_op(self, scope):
		"""
		Add Adam optimizer to maximize q loss.
		Args:
			scope (string): The network to train,
				typically "q" rather "target_q".
		"""
		optimizer = tf.keras.optimizers.Adam(self.lr)
		train_vars = tfv1.get_collection(
			tfv1.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
		self.train_op = optimizer.minimize(self.loss, train_vars)
	
	def build(self):
		"""
		Build the model by adding all necessary variables.
		"""
		# add placeholders
		self.s = tfv1.placeholder(
			tf.float32, (None, self.observation_dim), name="state")
		self.a = tfv1.placeholder(
			tf.float32, (None, self.action_dim), name="action")
		self.r = tfv1.placeholder(
			tf.float32, (None,), name="reward", )
		self.sp = tfv1.placeholder(
			tf.float32, (None, self.observation_dim), name="next_state")
		self.done_mask = tfv1.placeholder(
			tf.bool, (None,), name="done_mask")
		# compute deterministic action for the state
		self.actor = Actor(self.observation_dim, self.action_dim)
		self.mu = self.get_action_op(self.s, "mu")
		self.mup = self.get_action_op(self.sp, "mu", reuse=True)
		# compute target action for the next state (to update critic)
		self.target_mu = self.get_action_op(self.sp, "target_mu")
		# build critic (the q value function)
		self.critic = Critic(
			self.config,
			self.s,
			self.sp,
			self.mu,
			self.target_mu,
			self.mup,
			self.r,
			self.done_mask)
		# add update operator for target network (actor)
		self.add_soft_target_update_op("mu", "target_mu")
		self.add_hard_target_update_op("mu", "target_mu")
		# add loss
		self.add_loss_op(self.critic.qm)
		# add optimizer
		self.add_optimizer_op("mu")
	
	def initialize(self):
		"""
		Assumes the graph has been constructed (have called self.build_actor())
		Creates a tf Session and run initializer of variables
		"""
		# setting the seed
		# pdb.set_trace()
		
		# create tf session
		os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
		
		gpu_options = tfv1.GPUOptions(per_process_gpu_memory_fraction=0.7)
		gpu_config = tfv1.ConfigProto(gpu_options=gpu_options)
		gpu_config.gpu_options.allow_growth = True
		self.sess = tfv1.Session(config=gpu_config)
		
		# tensorboard stuff
		# self.add_summary()
		
		# initialize all variables
		init = tfv1.global_variables_initializer()
		self.sess.run(init)
		
		# let the target network the same as eval network
		self.sess.run(self.hard_update_target_op)
		
		# set critic's session
		self.critic.set_session(self.sess)
		
		# for saving network weights
		self.saver = tfv1.train.Saver()
	
	def choose_action(self, state):
		"""
		Given state, choose a deterministic action by mu
		"""
		return self.sess.run(self.mu, feed_dict={self.s: state[np.newaxis, :]})
	
	def train(self):
		"""
		Performs training of Actor & Critic
		"""
		# initialize replay buffer and variables
		replay_buffer = ReplayBuffer(self.config.buffer_size)
		
		t = last_eval = last_record = 0
		scores_eval = []  # list of scores computed at iteration time
		scores_eval += [self.evaluate()]
		
		prog = Progbar(target=self.config.nsteps_train)
		
		# interact with environment
		Noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_dim))
		while t < self.config.nsteps_train:
			# begin a new episode
			state = self.env.reset()
			epi_len = 0
			total_reward = 0
			Noise.reset()
			while epi_len < self.config.max_ep_len:
				t += 1
				last_eval += 1
				last_record += 1
				epi_len += 1
				# exploratory action with noise
				action = self.choose_action(state) + Noise()
				# perform action in env
				new_state, reward, done, info = self.env.step(action)
				# replay buffer stuff
				replay_buffer.store_transition(
					state, action, reward, new_state, done)
				
				state = new_state
				if (t > self.config.learning_start):
					# perform a training step
					q_eval = self.train_step(t, replay_buffer)
					
					# logging stuff
					if ((t % self.config.log_freq == 0) and (t % self.config.learning_freq == 0)):
						prog.update(t + 1, values=[("Q(s, mu(s))", q_eval)])
				
				# t <= learning start
				elif (t % self.config.log_freq == 0):
					sys.stdout.write("\rPopulating the memory {}/{}...".
									 format(t, self.config.learning_start))
					sys.stdout.flush()
				
				# count reward
				total_reward += reward
				if done or t >= self.config.nsteps_train:
					break
			
			if (t > self.config.learning_start) and (last_eval > self.config.eval_freq):
				# evaluate our policy
				last_eval = 0
				print("")
				scores_eval += [self.evaluate()]
			
			if (t > self.config.learning_start) and self.config.record \
					and (last_record > self.config.eval_freq):
				# record a episode (by video)
				last_record = 0
				self.record()
		
		# last words
		self.logger.info("- Training done.")
		self.save()
		scores_eval += [self.evaluate()]
		export_plot(scores_eval, "Scores", self.config.plot_output)
	
	def train_step(self, t, replay_buffer):
		"""
		Perform training step

		Args:
			t: (int) nths step
			replay_buffer: buffer for sampling
		"""
		q_eval = 0
		
		# perform training step
		if (t % self.config.learning_freq == 0):
			q_eval = self.learn_step(t, replay_buffer)
			# apply soft update
			self.sess.run(self.update_target_op)
		
		# occasionally save the weights
		if (t % self.config.saving_freq == 0):
			self.save()
		
		return q_eval
	
	def learn_step(self, t, replay_buffer):
		"""
		Performs an update of parameters by sampling from replay_buffer

		Args:
			t: number of iteration (episode and move)
			replay_buffer: ReplayBuffer instance .sample() gives batches
		Returns:
			loss: Q(s_t, mu(s_t))
		"""
		s_batch, a_batch, r_batch, sp_batch, done_mask_batch = \
			replay_buffer.sample(self.config.batch_size)
		
		fd = {
			# input
			self.s: s_batch,
			self.a: a_batch,
			self.r: r_batch,
			self.sp: sp_batch,
			self.done_mask: done_mask_batch,
			# maybe extra info
		}
		
		q_eval, _ = self.sess.run([-self.loss, self.train_op], feed_dict=fd)
		return q_eval
	
	def save(self):
		"""
		Save session
		"""
		if not os.path.exists(self.config.model_output):
			os.makedirs(self.config.model_output)
		
		self.saver.save(self.sess, self.config.model_output)
	
	def evaluate(self, env=None, num_episodes=None):
		"""
        Evaluation with same procedure as the training
        """
		# log our activity only if default call
		if num_episodes is None:
			self.logger.info("Evaluating...")
			num_episodes = self.config.num_episodes_test
		
		if env is None:
			env = self.env
		
		rewards = []
		
		for i in range(num_episodes):
			total_reward = 0
			state = env.reset()
			epi_len = 0
			while epi_len < self.config.max_ep_len:
				action = self.choose_action(state)
				# perform action in env
				new_state, reward, done, info = env.step(action)
				state = new_state
				# count reward
				total_reward += reward
				if done:
					break
			# updates to perform at the end of an episode
			rewards.append(total_reward)
		
		avg_reward = np.mean(rewards)
		sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
		
		if num_episodes > 1:
			# indicates this is not recording
			msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
			self.logger.info(msg)
		
		return avg_reward
	
	def record(self):
		"""
        Recreate an env and record a video for one episode
        """
		self.logger.info("Recording...")
		env = gym.make(self.config.env_name)
		env = gym.wrappers.Monitor(env, self.config.record_path,
								   video_callable=lambda x: True,
								   resume=True)
		self.evaluate(env, 1)
	
	def run(self):
		"""
		Apply procedures of training for a PG.
		"""
		# initialize
		self.initialize()
		# record one game at the beginning
		if self.config.record:
			self.record()
		# model
		self.train()
		# record one game at the end
		if self.config.record:
			self.record()
