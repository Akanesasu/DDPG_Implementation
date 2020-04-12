import argparse
import numpy as np
import tensorflow as tf
import gym
from tensorflow import keras
from actor import DDPG

from config import config
import random

if __name__ == '__main__':
	# Initialization stuff
	tf.compat.v1.disable_eager_execution()
	
	# make env
	env = gym.make(config.env_name)
	
	# train model
	model = DDPG(env, config)
	model.run()
	
	