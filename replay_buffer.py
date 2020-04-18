import numpy as np
import random

class ReplayBuffer(object):
	"""
	Based from Berkeley's Assignment
	"""
	
	def __init__(self, capacity):
		"""This is a memory efficient implementation of the replay buffer.
		
		Parameters
		----------
		size: int
			Max number of transitions to store in the buffer. When the buffer
			overflows the old memories are dropped.
		"""
		self.capacity = capacity
		self.next_idx = 0
		self.num_in_buffer = 0
		
		self.obs = None
		self.action = None
		self.reward = None
		self.next_obs = None
		self.done = None
	
	def _encode_sample(self, idxes):
		obs_batch = np.concatenate([self.obs[idx] for idx in idxes], 0)
		act_batch = self.action[idxes]
		rew_batch = self.reward[idxes]
		next_obs_batch = np.concatenate([self.next_obs[idx] for idx in idxes], 0)
		done_mask = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)
		
		return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask
	
	def sample(self, batch_size):
		"""Sample `batch_size` different transitions.

		i-th sample transition is the following:

		when observing `obs_batch[i]`, action `act_batch[i]` was taken,
		after which reward `rew_batch[i]` was received and subsequent
		observation  next_obs_batch[i] was observed, unless the epsiode
		was done which is represented by `done_mask[i]` which is equal
		to 1 if episode has ended as a result of that action.

		Parameters
		----------
		batch_size: int
			How many transitions to sample.

		Returns
		-------
		obs_batch: np.array
			Array of shape (batch_size, observation_dim) and dtype np.float32
		act_batch: np.array
			Array of shape (batch_size, action_dim) and dtype np.float32
		rew_batch: np.array
			Array of shape (batch_size,) and dtype np.float32
		next_obs_batch: np.array
			Array of shape (batch_size, observation_dim) and dtype np.float32
		done_mask: np.array
			Array of shape (batch_size,) and dtype np.float32
		"""
		idxes = random.sample(range(0, self.num_in_buffer - 1), batch_size)
		return self._encode_sample(idxes)
	
	def store_transition(self, obs, action, reward, next_obs, done):
		"""Store a single transition in the buffer at the next available index, overwriting
		old frames if necessary.

		Parameters
		----------
		obs: np.array
			Array of shape (observation_dim, ) and dtype np.float32
		action: np.array
			Array of shape (action_dim, ) and dtype np.float32
		reward: np.array
			Array of shape (1,) and dtype np.float32
		next_obs: np.array
			Array of shape (observation_dim, ) and dtype np.float32
		done: np.array
			Array of shape (1,) and dtype np.float32

		Returns
		-------
		idx: int
			Index at which the transition is stored.
		"""
		
		if self.obs is None:
			self.obs = np.empty([self.capacity], dtype=np.float32)
			self.action = np.empty([self.capacity], dtype=np.float32)
			self.reward = np.empty([self.capacity], dtype=np.float32)
			self.next_obs = np.empty([self.capacity], dtype=np.float32)
			self.done = np.empty([self.capacity], dtype=np.float32)
		
		self.obs[self.next_idx] = obs
		self.action[self.next_idx] = action
		self.reward[self.next_idx] = reward
		self.next_obs[self.next_idx] = next_obs
		self.done[self.next_idx] = done
		
		ret = self.next_idx
		self.next_idx = (self.next_idx + 1) % self.capacity
		self.num_in_buffer = min(self.capacity, self.num_in_buffer + 1)
		
		return ret
