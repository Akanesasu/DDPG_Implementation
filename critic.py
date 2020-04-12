import tensorflow as tf
import tensorflow.compat.v1 as tfv1
from tensorflow.keras import regularizers
import numpy as np


class Critic(object):
	def __init__(self, config, state, next_state,
				 action, next_action, reward, done_mask):
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
		self.ap = next_action
		self.r = reward
		self.done_mask = done_mask

		self.lr = config.critic_learning_rate
		self.weight_decay = config.weight_decay

	def set_session(self, session):
		# the same session as actor
		self.sess = session

	def get_q_value_op(self, state, action, scope):
		"""
		Returns Q value for (state, action) pair
		Args:
				state, action : the pair to evaluate q
				scope: indicate whether is target network or not
		Returns:
				A scalar tf tensor denoting Q(state, action)
		"""
		with tfv1.variable_scope(scope):
			out = tf.keras.layers.Dense(400, activation=tf.nn.relu,
				kernel_regularizer=regularizers.l2(self.weight_decay))(state)
			out = tf.keras.layers.concatenate([out, action], axis=1)
			out = tf.keras.layers.Dense(300, activation=tf.nn.relu,
				kernel_regularizer=regularizers.l2(self.weight_decay))(out)
			out = tf.keras.layers.Dense(1,
				kernel_regularizer=regularizers.l2(self.weight_decay))(out)
		return out

	def add_loss_op(self, q, target_q, r, done_mask):
		not_done = 1 - tf.cast(done_mask, tf.float32)
		Target = r + self.config.gamma * not_done * target_q
		self.loss = tf.losses.mean_squared_error(Target, q)

	def add_optimizer_op(self, scope):
		optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
		train_vars = tfv1.get_collection(tfv1.GraphKeys.TRAINABLE_VARIABLES,
										 scope=scope)
		self.train_op = optimizer.minimize(self.loss, train_vars)

	def add_soft_target_update_op(self, q_scope, target_q_scope):
		q_vars = tfv1.get_collection(tfv1.GraphKeys.TRAINABLE_VARIABLES,
		                             scope=q_scope)
		tar_vars = tfv1.get_collection(tfv1.GraphKeys.TRAINABLE_VARIABLES,
		                             scope=target_q_scope)
	
	def build_critic(self):
		# compute Q(s, mu(s)) for policy gradient
		self.q = self.get_q_value_op(self.s, self.a, "q")
		# compute Q(s', a' = mu(s')) for bootstrapping of critic
		self.target_q = self.get_q_value_op(self.sp, self.ap, "target_q")
		# add loss
		self.add_loss_op(self.q, self.target_q, self.r, self.done_mask)
		# add training op
		self.add_optimizer_op()

	def update_critic(self, states, actions, q_targets):
		self.sess.run(self.update_q_op, feed_dict={self.s})
