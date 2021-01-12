# Contains all dependencies used in bot
# First file loaded

from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
import math, random
import numpy as np
import scipy.optimize
import scipy.ndimage
import scipy.signal
from queue import PriorityQueue
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Global constants
INF = 999999999999

# All game state goes here - everything, even mundane
state = {}

# Bot training weights
# 0 - shipyard reward
# 1 - mine reward
# 2 - attack weights
# 3 - return weights
# 4 - spawn weights
# 5 - guard weights
# 6 - navigation weights
# 7 - control weights

temp = []
weights = weights.split("\n")
for line in weights:
    temp.append(np.array(list(map(float, line.split()))))
weights = temp
