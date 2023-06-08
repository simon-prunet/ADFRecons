import numpy as np
from recons import antenna_set
import sys
import os
from recons import coincidence_set
c_light = 2.997924580e8


class Make_mask:
	def __init__(self, antenna_position_file, coinc_table_file,  masked_output_file = 'Rec_coinctable_simulated_masked.txt' ):
		self.antennas = antenna_set(antenna_position_file)
		self.coincidence_set = coincidence_set(coinc_table_file, self.antennas)
		self.masked_output_file = masked_output_file
        
        
        
        
	def write_masked_coincidence_file(self, triggered_an_index,triggered_event_id, triggered_peak_time, triggered_amp_array, masked_output_file):
    #(self, triggered_an_index,triggered_event_id, triggered_peak_time, triggered_amp_array, masked_output_file)
		f = open(masked_output_file, 'w')
		#f.write("%12.0le %12.0le %12.10le %12.10le \n"%(triggered_an_index,triggered_event_id, triggered_peak_time, triggered_amp_array ))
		#this one works but not the way I would like it to be :D
		#f.write("{} {} {} {}\n".format(triggered_an_index, triggered_event_id, triggered_peak_time,triggered_amp_array))
		#np.savetxt(f,  np.transpose([triggered_an_index.astype(int), triggered_event_id.astype(int), triggered_peak_time.astype(float), triggered_amp_array.astype(float)]))
		
		
		for i in range(len(triggered_an_index)):
			f.write("{0:<4d} {1:<10d} {2:<12.10g} {3:<12.10g}\n".format(triggered_an_index[i], triggered_event_id[i],triggered_peak_time[i]/c_light, triggered_amp_array[i] ))
		f.close()
		
        
        
	def __call__(self, shower_center = [0,0], r = 6000, number_of_sets = 100):
		#number_of_sets is the same as in simpulation input
		self.shower_center = shower_center
		self.r = r
		self.number_of_sets = number_of_sets


		mask = np.zeros((self.antennas.nants), dtype = bool)
        
		for i in range(self.antennas.nants):
			distance = np.sqrt((self.antennas.coordinates[i,0] - shower_center[0])**2 + (self.antennas.coordinates[i,1] - shower_center[1])**2)
			if distance <= r:
				mask[i] = True
			else:
				mask[i] = False

		triggered_an_index = self.coincidence_set.antenna_index_array[:, mask]
		triggered_event_id = self.coincidence_set.coinc_index_array[:, mask]
		triggered_peak_time = self.coincidence_set.peak_time_array[:, mask]
		triggered_amp_array = self.coincidence_set.peak_amp_array[:, mask]
        
		triggered_an_index = triggered_an_index.flatten()
		triggered_event_id = triggered_event_id.flatten()
		triggered_peak_time = triggered_peak_time.flatten()
		triggered_amp_array = triggered_amp_array.flatten()
		#params = np.array([triggered_an_index.astype(int),triggered_event_id.astype(int), triggered_peak_time.astype(float), triggered_amp_array.astype(float)]).T



		self.write_masked_coincidence_file(triggered_an_index.astype(int),triggered_event_id.astype(int), triggered_peak_time.astype(float), triggered_amp_array.astype(float), self.masked_output_file)
		#print(params)


		return 

        


