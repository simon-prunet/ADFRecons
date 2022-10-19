import numpy as np
#import wavefronts as wa
import sys
import os

class antenna_set:

    def __init__(self, position_file):

        if (not os.path.exists(position_file)):
            print("Antenna coordinates file %s does not exist"%position_file)
            return
        print(" Reading antenna positions from file %s"%position_file)
        self.position_input_file = position_file
        self.indices = np.loadtxt(self.position_input_file, usecols=(0,), dtype=int)
        self.coordinates = np.loadtxt(self.position_input_file, usecols = (1,2,3))
        self.init_ant = np.min(self.indices)
        self.nants = np.size(indices)
        return

class coincidence_set:

    def __init__(self, coinc_table_file):

        if (not os.path.exists(coinc_table_file)):
            print("Coincidence table file %s does not exist"%coinc_table_file)
            return

        self.coinc_table_file = coinc_table_file
        print(" Reading coincidence(s): index, peak time, peak amplitude from file %s"%self.coinc_table_file)
        tmp = np.loadtxt(self.coinc_table_file,dtype='int',usecols=(0,1))
        antenna_index_array = tmp[:,0]
        coinc_index_array   = tmp[:,1]
        tmp2 = np.loadtxt(self.coinc_table_file,usecols=(2,3)) # floats
        peak_time_array = tmp2[:,0]
        peak_amp_array  = tmp2[:,1]
        coinc_indices = np.unique(coinc_index_array)
        ncoincs = len(coinc_indices)

        # Filter out coincidences with small number of antennas
        self.ncoincs = 0
        for index in coinc_indices:
            current_mask = (coinc_index_array==index)
            current_length = np.sum(current_mask)
            if current_length>3:
                self.nants = current_length
                self.ncoincs += 1

        print(self.nants,self.ncoincs)
        # Now create the structure and populate it
        self.antenna_index_array = np.zeros((self.ncoincs,self.nants),dtype='int')
        self.coinc_index_array   = np.zeros((self.ncoincs,self.nants),dtype='int')
        self.peak_time_array     = np.zeros((self.ncoincs,self.nants))
        self.peak_amp_array      = np.zeros((self.ncoincs,self.nants))
        # Filter and read
        current_coinc = 0
        for index in coinc_indices:
            mask = (coinc_index_array==index)
            print (mask)
            current_length = np.sum(mask)
            if current_length>3:
                self.antenna_index_array[current_coinc,:] = antenna_index_array[mask]
                self.coinc_index_array[current_coinc,:] = coinc_index_array[mask]
                self.peak_time_array[current_coinc,:] = peak_time_array[mask]
                self.peak_amp_array[current_coinc,:] = peak_amp_array[mask]
                current_coinc += 1
        return

def 



def main():

    if (len(sys.argv) != 3):
        print ("Usage: python recons.py <recons_type> <data_dir> ")
        print ("recons_type = 0 (plane wave), 1 (spherical wave), 2 (ADF)")
        print ("data_dir is the directory containing the coincidence files")
        sys.exit(1)

    recons_type = sys.argv[1]
    data_dir = sys.argv[2]

    print('recons_type = ',recons_type)
    print('data_dir = ',data_dir)

    # Read antennas indices and coordinates
    an = antenna_set(data_dir+'/coord_antennas.txt')
    # Read coincidences (antenna index, coincidence index, peak time, peak amplitude)
    # Routine only keep valid number of antennas (>3)
    co = coincidence_set(data_dir+'/Rec_coinctable.txt')
    


    return

if __name__ == '__main__':

    main()

