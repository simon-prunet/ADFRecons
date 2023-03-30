import numpy as np
from wavefronts import *
from recons import antenna_set
import sys
import os
import scipy.optimize as so
import numdifftools as nd
c_light = 2.997924580e8


class Simulations:

		def __init__(self,antenna_position_file):

				self.antennas = antenna_set(antenna_position_file)
				