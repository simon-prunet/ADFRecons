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
	
	def write_coincidence_file(self,tants,coincidence_output_file):

		# Set seed to produce the same (random) integer event ids
		np.random.seed(1)

		f = open(coincidence_output_file,'w')
		for i in range(self.number_of_events*self.number_of_sets):
			event_id = np.random.random_integers(max_id) # Create a random number < max_id
			for j in range(self.antennas.nants):
				f.write("{0:<4d} {1:<10d} {2:<12.10g} {3:<12.10g}\n".format(j,event_id,tants[i,j]/c_light,np.nan))
		f.close()
		return

	def __call__(self,params,simulation_type='PWF',iseed=1234, sigma_t = 5e-9, number_of_events=1, number_of_sets=1):
     		#number_of_sets stands for the number of sets of different parameters. For each set of parameters "number_of_events"- events are computed
			#for the case of number_of_sets=1 the initial code must be used, or separate case added to this code
		self.number_of_events = number_of_events
		self.number_of_sets = number_of_sets
		self.simulation_type = simulation_type

		tants = np.zeros((number_of_events*number_of_sets,self.antennas.nants))
		a=0
		if (self.simulation_type=='PWF'):
			for i in range(number_of_sets):
				for k in range(number_of_events):                       
					tants[a,:] = PWF_simulation(params[i,:], self.antennas.coordinates, iseed=iseed, sigma_t=sigma_t)
					a+=1
					
		elif (self.simulation_type=='SWF'):
			for i in range(number_of_sets):
				for k in range(number_of_events):
					tants[a,:] = SWF_simulation(params[i,:], self.antennas.coordinates, iseed=iseed, sigma_t=sigma_t)
					a+=1
		else:
			print('Only PWF (plane wave) and SWF (spherical wave) are supported now')
			return

		self.write_coincidence_file(tants,self.coincidence_output_file)
		return
  
  
  
  		

