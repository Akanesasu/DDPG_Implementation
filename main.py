import argparse
import numpy as np
import gym
from DDPG import DDPG
from config import config
import random

if __name__ == '__main__':
	# Initialization stuff
	#
	
	# make env
	env = gym.make(config.env_name)
	
	# train model
	model = DDPG(env, config)
	model.run()
	
	