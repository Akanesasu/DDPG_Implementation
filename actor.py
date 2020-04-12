import os
import gym
import numpy as np
import sys
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
from tensorflow.keras.layers import Dense
from keras.utils import Progbar
from critic import Critic


class DDPG(object):
	"""
	Abstract Class for implementing Deep Deterministic Policy Gradient
	"""

	def __init__(self, env, config):
		"""
		Initialize Policy Gradient Class
		Args:
						env: an OpenAI Gym environment
						config: class with hyperparameters
		"""
		# store hyperparameters
		self.config = config
		self.r_seed = config.r_seed
		self.env = env
		self.env.seed(self.r_seed)

		self.observation_dim = self.env.observation_space.shape[0]
		self.action_dim = self.env.action_space.shape[0]

		self.lr = config.actor_learning_rate
		self.tau = config.tau

		# build model
		self.build_actor()

	def add_placeholders_op(self):
		# add placeholders
		self.s = tfv1.placeholder(
			tf.float32, (None, self.observation_dim), name="state")
		self.a = tfv1.placeholder(
			tf.float32, (None, self.action_dim), name="action")
		self.r = tfv1.placeholder(tf.float32, (None,), name="reward", )
		self.sp = tfv1.placeholder(
			tf.float32, (None, self.observation_dim), name="next_state")
		self.done_mask = tfv1.placeholder(tf.bool, (None,), name="done_mask")

	def get_action_op(self, state, scope):
		"""
		Return deterministic action given the state
		Args:
				state: (tf tensor)
						shape = (batch_size, observation_dim)
				scope: (string)
						scope name that specifies if target network or not
		"""
		out = state
		with tfv1.variable_scope(scope):
			out = tf.keras.layers.Dense(400, activation=tf.nn.relu)(out)
			out = tf.keras.layers.Dense(300, activation=tf.nn.relu)(out)
			out = tf.keras.layers.Dense(
				self.action_dim, activation=tf.nn.tanh)(out)
		return out

	def add_soft_target_update_op(self, a_scope, target_a_scope):
		"""
		Move target weights slowly toward current weights.
		"""
		mu_vars = tfv1.get_collection(tfv1.GraphKeys.TRAINABLE_VARIABLES,
									  scope=a_scope)
		tar_vars = tfv1.get_collection(tfv1.GraphKeys.TRAINABLE_VARIABLES,
									   scope=target_a_scope)

		upd_vars = [tar_vars[i] + self.tau * (mu_vars[i] - tar_vars[i])
					for i in range(len(mu_vars))]
		self.update_target_op = tfv1.group(*[tfv1.assign(tar_var, upd_var)
											 for tar_var, upd_var in zip(tar_vars, upd_vars)])

	def add_loss_op(self, q):
		"""
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
		

	def build_actor(self):
		"""
		Build the actor model by adding all necessary variables.
		"""
		# add placeholders
		self.add_placeholders_op()
		# compute deterministic action for the state
		self.mu = self.get_action_op(self.s, "mu")
		# compute target action for the next state (to update critic)
		self.target_mu = self.get_action_op(self.sp, "target_mu")
		# build critic (the q value function)
		self.critic = Critic(
			self.config,
			self.s,
			self.sp,
			self.mu,
			self.target_mu)
		# add update operator for target network (actor)
		self.add_soft_target_update_op("mu", "target_mu")
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
		self.sess = tfv1.Session()

		# tensorboard stuff
		# self.add_summary()

		# initialize all variables
		init = tfv1.global_variables_initializer()
		self.sess.run(init)

		self.critic.set_session(self.sess)

	def train(self):
		"""
		To be added.
		"""
		pass

	def evaluate(self):
		"""
		To be added.
		"""
		pass

	def record(self):
		"""
		To be added.
		"""
		pass

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
