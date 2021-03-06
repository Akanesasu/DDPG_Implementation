import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from config import config


class Policy_Network(nn.Module):
	def __init__(self, name, state_dim, action_dim, init_w=3e-3):
		super(Policy_Network, self).__init__()
		self.name = name
		self.fc1 = nn.Linear(state_dim, 128)
		self.fc2 = nn.Linear(128, 64)
		self.fc3 = nn.Linear(64, action_dim)
		# initialize weights and biases of last layer as in DDPG paper
		nn.init.uniform_(self.fc3.weight, -init_w, init_w)
		nn.init.uniform_(self.fc3.bias, -init_w, init_w)
	
	def forward(self, state):
		"""
		Return deterministic action given the state
		Args:
			state: (torch tensor)
				shape = (batch_size, state_dim)
		"""
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		x = torch.tanh(self.fc3(x))
		return x


class Actor(object):
	"""
	Class for actor in DDPG
	"""
	
	def __init__(self, state_dim, action_dim, device=None):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self._build_actor()
		# assign device
		self.device = device
		if self.device is None:
			self.device = torch.device("cuda" if torch.cuda.is_available()
									   else "cpu")
	
	def _build_actor(self):
		self.mu = Policy_Network(
			"mu", self.state_dim, self.action_dim)
		self.target_mu = Policy_Network(
			"target_mu", self.state_dim, self.action_dim)
		# let the value of the target net equal to eval net
		self._hard_target_update(self.mu, self.target_mu)
		# add optimizer
		self.optimizer = optim.Adam(self.mu.parameters(), lr=config.actor_lr)
	
	def soft_target_update(self, eval_net=None, target_net=None):
		if eval_net is None:
			eval_net = self.mu
		if target_net is None:
			target_net = self.target_mu
		eval_par = eval_net.state_dict()
		tar_par = target_net.state_dict()
		upd_par = {}  # dict of updated parameters
		for par in eval_par:
			# par here is a string like "fc1.weight"
			upd_par[par] = tar_par[par] + \
						   config.tau * (eval_par[par] - tar_par[par])
		target_net.load_state_dict(upd_par)
	
	def _hard_target_update(self, eval_net, target_net):
		target_net.load_state_dict(eval_net.state_dict())
	
	def train_on_batch(self, s, q_net):
		"""
		Train by gradient ascent (formula without minibatch):
				θ = θ + ∇_a(Q(s_t, a)|a=μ(s_t)) * ∇_θ(μ(s_t|θ))
				which is equivalent to (by chain rule of derivative)
				TODO: (really?)
				θ = θ + ∇_θ(Q(s_t, μ(s_t|θ)))
				So we use -Q(s_t, μ(s_t)) as loss. (minimize it)
		Args:
			s (tf tensor): shape = (batch_size, state_dim)
			q_net (nn.Module): a callable critic
		Return:
			value of loss
		"""
		a = self.mu(s)
		loss = torch.mean(-q_net(s, a))
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return loss.item() # -Q(s, mu(s))
	
	def choose_action(self, s):
		"""
		Just used to choose, not used to train
		Args:
			s (torch tensor):
				shape = (state_dim)
		"""
		return self.mu(s).detach()
