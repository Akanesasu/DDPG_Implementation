import tensorflow as tf
import tensorflow.compat.v1 as tfv1
from tensorflow.keras import regularizers
import numpy as np


class Critic(object):
	def __init__(self, config, state, next_state,
				 action, action_by_mu, action_next, reward, done_mask):
		"""
		Initialize Critic.
		Args:
						config: global config
						state:  state placeholder
						next_state: next state placeholder
						action: (tf tensor) action given by actor
						next_action: (tf tensor) action of next state given by actor
		"""
		self.config = config
		self.s = state
		self.sp = next_state
		self.a = action
		self.mu = action_by_mu  # computed by actor
		self.mup = action_next  # computed by actor
		self.r = reward
		self.done_mask = done_mask

		self.lr = config.critic_learning_rate
		self.weight_decay = config.weight_decay
		self.tau = config.tau

	def set_session(self, session):
		# the same session as actor
		self.sess = session

	def get_q_value_op(self, state, action, scope, reuse=False):
		"""
		Returns Q value for (state, action) pairs
		Args:
						state, action (tensor) : the pair to be evaluated
						scope (string): indicate whether is target network or not
						reuse (bool): reuse of variables in the scope
		Returns:
						A scalar tf tensor denoting Q(state, action)
		"""
		with tfv1.variable_scope(scope, reuse=reuse):
			out = tf.keras.layers.Dense(400, activation=tf.nn.relu,
										kernel_regularizer=regularizers.l2(self.weight_decay))(state)
			out = tf.keras.layers.concatenate([out, action], axis=1)
			out = tf.keras.layers.Dense(300, activation=tf.nn.relu,
										kernel_regularizer=regularizers.l2(self.weight_decay))(out)
			out = tf.keras.layers.Dense(1,
										kernel_regularizer=regularizers.l2(self.weight_decay))(out)
		return out

	def add_loss_op(self, q, target_q, r, done_mask):
		"""
		Add mean squared error loss for a batch.
		Args:
				q, target_q: Q values computed by current network and target network.
				r, done_mask (Tensor): denote reward and whether done respectively.
		"""
		not_done = 1 - tf.cast(done_mask, tf.float32)
		Target = r + self.config.gamma * not_done * target_q
		self.loss = tf.losses.mean_squared_error(Target, q)

	def add_optimizer_op(self, scope):
		"""
		Add Adam optimizer to minimize MSE loss.
		Args:
				scope (string): The network to train,
						typically "q" rather "target_q".
		"""
		optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
		train_vars = tfv1.get_collection(tfv1.GraphKeys.TRAINABLE_VARIABLES,
										 scope=scope)
		self.train_op = optimizer.minimize(self.loss, train_vars)

	def add_soft_target_update_op(self, q_scope, target_q_scope):
		"""
		Move target weights slowly toward current weights.
		"""
		q_vars = tfv1.get_collection(tfv1.GraphKeys.TRAINABLE_VARIABLES,
									 scope=q_scope)
		tar_vars = tfv1.get_collection(tfv1.GraphKeys.TRAINABLE_VARIABLES,
									   scope=target_q_scope)

		upd_vars = [tar_vars[i] + self.tau * (q_vars[i] - tar_vars[i])
					for i in range(len(q_vars))]
		self.update_target_op = tfv1.group(*[tfv1.assign(tar_var, upd_var)
											 for tar_var, upd_var in zip(tar_vars, upd_vars)])

	def build_critic(self):
		"""
		Build the critic model by adding all necessary variables.
		"""
		# compute Q(s, a) for bootstrapping of critic
		self.q = self.get_q_value_op(self.s, self.a, "q")
		# compute Q(s', mu(s')) also for bootstrapping of critic
		self.target_q = self.get_q_value_op(self.sp, self.mup, "target_q")
		# compute Q(s, mu(s)) for computing policy gradient
		self.qm = self.get_q_value_op(self.s, self.mu, "q", reuse=True)
		# add soft target update op (critic)
		self.add_soft_target_update_op("q", "target_q")
		# add loss
		self.add_loss_op(self.q, self.target_q, self.r, self.done_mask)
		# add training op
		self.add_optimizer_op("q")
		
