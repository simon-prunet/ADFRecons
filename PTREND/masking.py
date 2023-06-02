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
        f = open(masked_output_file, 'a')
        f.write("%ld %ld %12.10le %12.10le \n"%(triggered_an_index,triggered_event_id, triggered_peak_time, triggered_amp_array ))


    def __call__(self, shower_center = [0,0], r = 2000, number_of_sets = 100):
        #number_of_sets is the same as in simpulation input
        self.shower_center = shower_center
        self.r = r
        self.number_of_sets = number_of_sets


        mask = list()
        for i in range(self.antennas.nants):
            distance = np.sqrt((self.antennas.coordinates[i,0] - shower_center[0])**2 + (self.antennas.coordinates[i,1] - shower_center[1])**2)
            if distance <= r:
                mask.append(True)
            else:
                mask.append(False)

        triggered_an_index = self.coincidence_set.antenna_index_array[mask*number_of_sets]
        triggered_event_id = self.coincidence_set.coinc_index_array[mask*number_of_sets]
        triggered_peak_time = self.coincidence_set.peak_time_array[mask*number_of_set]
        triggered_amp_array = self.coincidence_set.peak_amp_array[mask*number_of_sets]


        self.write_masked_coincidence_file(self, triggered_an_index,triggered_event_id, triggered_peak_time, triggered_amp_array, masked_output_file)

        return 

        


