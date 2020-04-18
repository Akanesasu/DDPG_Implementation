import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from config import config


class Q_Network(nn.Module):
	def __init__(self, name, state_dim, action_dim):
		super(Q_Network, self).__init__()
		self.name = name
		# Architecture same as DDPG paper (low dimensional feature version)
		self.fc1 = nn.Linear(state_dim, 400)
		self.fc2 = nn.Linear(400 + action_dim, 300)
		self.fc3 = nn.Linear(300, 1)
		# initialize weights and biases of last layer as in DDPG paper
		nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
		nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)
	
	def forward(self, s, a):
		"""
		Args:
			s (torch tensor):
				shape = (batch_size, state_dim)
			a (torch tensor):
				shape = (batch_size, action_dim)
		"""
		x = F.relu(self.fc1(s))
		x = torch.cat((x, a), 1)
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		# x is a scalar denoting Q(s,a)
		return x


class Critic(object):
	"""
	Class for critic in DDPG
	"""
	
	def __init__(self, state_dim, action_dim, device=None):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self._build_critic()
		# assign device
		self.device = device
		if self.device is None:
			self.device = torch.device("cuda" if torch.cuda.is_available()
									   else "cpu")
	
	def _build_critic(self):
		self.q = Q_Network("q", self.state_dim, self.action_dim)
		self.target_q = Q_Network("target_q", self.state_dim, self.action_dim)
		# let the value of the target net equal to eval net
		self._hard_target_update(self.q, self.target_q)
		# add optimizer
		self.optimizer = optim.Adam(self.q.parameters(), lr=config.critic_lr,
									weight_decay=config.weight_decay)
	
	def _soft_target_update(self, eval_net=None, target_net=None):
		if eval_net is None:
			eval_net = self.q
		if target_net is None:
			target_net = self.target_q
		eval_par = eval_net.state_dict()
		tar_par = target_net.state_dict()
		upd_par = {}  # updated parameters
		for par in eval_par:
			# par here is a string like "fc1.weight"
			upd_par[par] = tar_par[par] + \
						   config.tau * (eval_par[par] - tar_par[par])
		target_net.load_state_dict(upd_par)
	
	def _hard_target_update(self, eval_net, target_net):
		target_net.load_state_dict(eval_net.state_dict())
	
	def train_on_batch(self, s, a, sp, mup, r, done):
		"""
		Train by bootstrapping
		Loss: (reward + gamma * Q(s', mu'(s')) - Q(s, a)) ^ 2
		Args:
			s:	shape = (batch_size, state_dim)
			a:	shape = (batch_size, action_dim)
			sp: s'
				shape = (batch_size, state_dim)
			mup: mu'(s') which is computed by actor
				shape = (batch_size, action_dim)
		"""
		assert s.shape[0] == config.batch_size
		not_done = 1 - done
		target_q_values = r + config.gamma * not_done * \
						  self.target_q(sp, mup).detach()
		q_values = self.q(s, a)
		loss = F.mse_loss(q_values, target_q_values)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		# move target network toward eval network
		self._soft_target_update()
