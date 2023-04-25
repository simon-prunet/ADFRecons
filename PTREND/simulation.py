import numpy as np
from wavefronts import *
from recons import antenna_set
import sys
import os
import scipy.optimize as so
import numdifftools as nd
c_light = 2.997924580e8
max_id = 1000000000

class Simulations:

	def __init__(self,antenna_position_file,coincidence_output_file='Rec_coinctable_simulated.txt'):

		self.antennas = antenna_set(antenna_position_file)
		self.coincidence_output_file=coincidence_output_file
	
	def write_coincidence_file(self,tants,coincidence_output_file, Aants=None):

		# Set seed to produce the same (random) integer event ids
		np.random.seed(1)

		f = open(coincidence_output_file,'w')
		for i in range(self.number_of_events):
			event_id = np.random.random_integers(max_id) # Create a random number < max_id
			if (Aants==None):
				for j in range(self.antennas.nants):
					f.write("{0:<4d} {1:<10d} {2:<12.10g} {3:<12.10g}\n".format(j,event_id,tants[i,j]/c_light,np.nan))
			else:
				for j in range(self.antennas.nants):
					f.write("{0:<4d} {1:<10d} {2:<12.10g} {3:<12.10g}\n".format(j,event_id,tants[i,j]/c_light,Aants[i,j]))
		f.close()
		return

	def __call__(self,params,simulation_type='PWF',iseed=1234, sigma_t = 5e-9, number_of_events=1):

		self.number_of_events = number_of_events
		self.simulation_type = simulation_type

		tants = np.zeros((number_of_events,self.antennas.nants))
		if (self.simulation_type=='PWF'):
			tants[0,:] = PWF_simulation(params, self.antennas.coordinates, iseed=iseed, sigma_t=sigma_t)
			for i in range(1, number_of_events):
				tants[i,:] = PWF_simulation(params, self.antennas.coordinates, sigma_t=sigma_t)
		elif (self.simulation_type=='SWF'):
			tants[0,:] = SWF_simulation(params, self.antennas.coordinates, iseed=iseed, sigma_t=sigma_t)
			for i in range(1, number_of_events):
				tants[i,:] = SWF_simulation(params, self.antennas.coordinates,sigma_t=sigma_t)
		elif (self.simulation_type=='ADF'):
			Aants = np.zeros((number_of_events,self.antennas.nants))
			Aants[0,:] = SWF_simulation(params,self.antennas.coordinates,self.)
		else:
			print('Only PWF (plane wave) and SWF (spherical wave) are supported now')
			return

		self.write_coincidence_file(tants,self.coincidence_output_file)
		return






