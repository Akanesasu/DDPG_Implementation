import numpy as np
import random
from collections import namedtuple

Transition = namedtuple('Transition',
						('state', 'action', 'reward', 'next_state', 'done_mask'))


class ReplayBuffer(object):
	def __init__(self, capacity):
		"""
		Args:
			capacity: int
				Max number of transitions to store in the buffer. When the buffer
				overflows the old memories are dropped.
		"""
		self.capacity = capacity
		self.next_idx = 0
		self.memory = []
	
	def sample(self, batch_size):
		"""
		Sample `batch_size` different transitions.
		Args:
			batch_size: int
				How many transitions to sample.
		Return:
			Transitions (batch of s, batch of a, ...)
		"""
		transitions = random.sample(self.memory, batch_size)
		return zip(*transitions)
	
	def store_transition(self, state, action, reward, next_state, done):
		"""
		Store a single transition in the buffer at the next available index,
		overwriting old transitions if necessary.
		"""
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.next_idx] = Transition(state, action, reward, next_state, done)
		self.next_idx = (self.next_idx + 1) % self.capacity

