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
from config import config
from torch.utils.tensorboard import SummaryWriter

class DDPG(object):
	"""
	Abstract Class for implementing Deep Deterministic Policy Gradient
	"""
	
	def __init__(self, env, logger=None):
		"""
		Initialize Policy Gradient Class
		Args:
			env: an OpenAI Gym environment
			config: class with hyperparameters
		"""
		# directory for training outputs
		if not os.path.exists(config.output_path):
			os.makedirs(config.output_path)
		
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
	
	def save(self):
		"""
		Save model weights
		"""
		if not os.path.exists(config.model_output):
			os.makedirs(config.model_output)
		torch.save(self.actor.mu.state_dict(), config.model_output + "actor")
		torch.save(self.critic.q.state_dict(), config.model_output + "critic")
	
	def train(self):
		"""
		Performs training of Actor & Critic
		"""
		# initialize replay buffer
		replay_buffer = ReplayBuffer(config.buffer_size)
		
		t = last_eval = last_record = 0
		scores_eval = []  # list of scores computed at iteration time
		scores_eval += [self.evaluate()]
		
		# interact with environment
		Noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_dim))
		while t < config.nsteps_train:
			# begin a new episode
			state = self.env.reset()
			state = torch.tensor([state], device=config.device,
								 dtype=torch.float32)
			epi_len = 0
			total_reward = 0
			Noise.reset()
			while epi_len < config.max_ep_len:
				t += 1
				last_eval += 1
				last_record += 1
				epi_len += 1
				# action with exploratory noise
				action = (self.actor.choose_action(state) + Noise()).type(torch.float32)
				# perform action in env
				new_state, reward, done, info = self.env.step(action[0])
				new_state = torch.tensor([new_state], device=config.device,
										 dtype=torch.float32)
				reward = torch.tensor([[reward]], device=config.device)
				done = torch.tensor([[done]], device=config.device,
									dtype=torch.float32)
				# replay buffer stuff
				replay_buffer.store_transition(
					state, action, reward, new_state, done)
				
				state = new_state
				if (t > config.learning_start):
					# perform a training step
					q_eval, mse_eval = self.train_step(t, replay_buffer)
					# logging stuff
					if ((t % config.log_freq == 0) and (t % config.learning_freq == 0)):
						sys.stdout.write("\rStep:[{}/{}]\t actor_q={:.5f}\t critic_loss={:.5f}".
										 format(t, config.nsteps_train, q_eval, mse_eval))
						sys.stdout.flush()
				# t <= learning start
				elif (t % config.log_freq == 0):
					sys.stdout.write("\rPopulating the memory {}/{}...".
									 format(t, config.learning_start))
					sys.stdout.flush()
				
				# count reward
				total_reward += reward.item()
				if done.item() or t >= config.nsteps_train:
					break
			
			if (t > config.learning_start) and (last_eval > config.eval_freq):
				# evaluate our policy
				last_eval = 0
				print("")
				scores_eval += [self.evaluate()]
			
			if (t > config.learning_start) and config.record \
					and (last_record > config.eval_freq):
				# record a episode (by video)
				last_record = 0
				self.record()
		
		# last words
		self.logger.info("- Training done.")
		self.save()
		scores_eval += [self.evaluate()]
		export_plot(scores_eval, "Scores", config.plot_output)
	
	def train_step(self, t, replay_buffer):
		"""
		Perform training step
		Args:
			t: (int) number of iteration (episode and move)
			replay_buffer: ReplayBuffer instance .sample() gives batches
		Returns:
			-actor_loss, which is  Q(s_t, mu(s_t))
			critic_loss, which is MSE(Q(s', mu'(s')) + r - Q(s, a))
		"""
		q_eval = mse_eval = 0
		# perform training step
		if (t % config.learning_freq == 0):
			# Performs an update of parameters by sampling from replay_buffer
			s, a, r, sp, done = replay_buffer.sample(config.batch_size)
			s = torch.cat(s)
			a = torch.cat(a)
			r = torch.cat(r)
			sp = torch.cat(sp)
			done = torch.cat(done)
			# train the actor and record Q
			q_eval = -self.actor.train_on_batch(s, self.critic.q)
			# train the critic by bootstrapping
			mup = self.actor.target_mu(sp).detach()
			mse_eval = self.critic.train_on_batch(s, a, sp, mup, r, done)
			# move target network toward eval network
			self.actor.soft_target_update()
			self.critic.soft_target_update()
			
		# occasionally save the weights
		if (t % config.saving_freq == 0):
			self.save()
			
		return q_eval, mse_eval
	
	def evaluate(self, env=None, num_episodes=None):
		"""
        Evaluation with same procedure as the training
        """
		# log our activity only if default call
		if num_episodes is None:
			self.logger.info("Evaluating...")
			num_episodes = config.num_episodes_test
		
		if env is None:
			env = self.env
		
		rewards = []
		
		for i in range(num_episodes):
			total_reward = 0
			state = env.reset()
			state = torch.tensor([state], device=config.device,
								 dtype=torch.float32)
			epi_len = 0
			while epi_len < config.max_ep_len:
				action = self.actor.choose_action(state)
				# perform action in env
				new_state, reward, done, info = env.step(action[0])
				new_state = torch.tensor([new_state], device=config.device,
										 dtype=torch.float32)
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
		env = gym.make(config.env_name)
		env = gym.wrappers.Monitor(env, config.record_path,
								   video_callable=lambda x: True,
								   resume=True)
		self.evaluate(env, 1)
	
	def run(self):
		"""
		Apply procedures of training for a PG.
		"""
		# record one game at the beginning
		if config.record:
			self.record()
		# model
		self.train()
		# record one game at the end
		if config.record:
			self.record()
