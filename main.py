import argparse
import numpy as np
import tensorflow as tf
import gym
from tensorflow import keras
from actor import DDPG

from config import config
import random

if __name__ == '__main__':
	# make env
	env = gym.make(config.env_name)
	
	# train model
	model = DDPG(env, config)
	model.run()
	
	