import os
import gym
import numpy as np
import sys
from critic import Critic
from actor import Actor
from replay_buffer import ReplayBuffer
from OrnsteinUhlenbeck import OrnsteinUhlenbeckActionNoise
from general_utils import *
import torch
from torch.utils.tensorboard import SummaryWriter

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
			
		self.env = env
		self.env.seed(config.r_seed)
		
		self.state_dim = self.env.observation_space.shape[0]
		self.action_dim = self.env.action_space.shape[0]
		
		# add actor and critic
		# actor compute deterministic action for the state
		self.actor = Actor(self.state_dim, self.action_dim)
		# critic compute the q value function
		self.critic = Critic(self.state_dim, self.action_dim)
	
	def learn_step(self, t, replay_buffer):
		"""
		Performs an update of parameters by sampling from replay_buffer

		Args:
			t: number of iteration (episode and move)
			replay_buffer: ReplayBuffer instance .sample() gives batches
		Returns:
			-loss, which is  Q(s_t, mu(s_t))
		"""
		s, a, r, sp, done = replay_buffer.sample(self.config.batch_size)
		# train the actor and record loss
		q_eval = -self.actor.train_on_batch(s, self.critic.q)
		# train the critic by bootstrapping
		mup = self.actor.target_mu(sp).detach()
		self.critic.train_on_batch(s, a, sp, mup, r, done)
		return q_eval
	
	def train(self):
		"""
		Performs training of Actor & Critic
		"""
		# initialize replay buffer
		replay_buffer = ReplayBuffer(self.config.buffer_size)
		
		t = last_eval = last_record = 0
		scores_eval = []  # list of scores computed at iteration time
		scores_eval += [self.evaluate()]
		
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
				# action with exploratory noise
				action = self.actor.mu(state).item() + Noise()
				# perform action in env
				new_state, reward, done, info = self.env.step(action)
				done = np.array(done, dtype=np.float32)
				# replay buffer stuff
				replay_buffer.store_transition(
					state, action, reward, new_state, done)
				
				state = new_state
				if (t > self.config.learning_start):
					# perform a training step
					q_eval = self.train_step(t, replay_buffer)
					# logging stuff
					if ((t % self.config.log_freq == 0) and (t % self.config.learning_freq == 0)):
						pass
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
		# occasionally save the weights
		if (t % self.config.saving_freq == 0):
			self.save()
		return q_eval
	
	def save(self):
		"""
		Save model weights
		"""
		if not os.path.exists(self.config.model_output):
			os.makedirs(self.config.model_output)
		torch.save(self.actor.mu.state_dict(), self.config.model_output + "actor")
		torch.save(self.critic.q.state_dict(), self.config.model_output + "critic")
	
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
				action = self.actor.mu(state).item()
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
		# logging
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
		# record one game at the beginning
		if self.config.record:
			self.record()
		# model
		self.train()
		# record one game at the end
		if self.config.record:
			self.record()
