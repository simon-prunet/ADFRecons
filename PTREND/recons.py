import numpy as np
from wavefronts import *
import sys
import os
import scipy.optimize as so
c_light = 2.997924580e8

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
        self.nants = np.size(self.indices)
        return

class coincidence_set:

    def __init__(self, coinc_table_file, antenna_set_instance):

        if (not os.path.exists(coinc_table_file)):
            print("Coincidence table file %s does not exist"%coinc_table_file)
            return

        self.coinc_table_file = coinc_table_file
        # This assumes an antenna_set instance has been created first
        if (not isinstance(antenna_set_instance,antenna_set)):
            print("Usage: co = coincidence_set(coinc_table_file,antenna_set_instance)")
            print("where coinc_table_file is a coincidence table file, and")
            print("antenna_set_instance is an instance of the class antenna_set")
            return
        self.ant_set = antenna_set_instance
        print(" Reading coincidence(s): index, peak time, peak amplitude from file %s"%self.coinc_table_file)
        tmp = np.loadtxt(self.coinc_table_file,dtype='int',usecols=(0,1))
        antenna_index_array = tmp[:,0]
        coinc_index_array   = tmp[:,1]
        tmp2 = np.loadtxt(self.coinc_table_file,usecols=(2,3)) # floats
        peak_time_array = tmp2[:,0]*c_light
        peak_amp_array  = tmp2[:,1]
        coinc_indices = np.unique(coinc_index_array)
        ncoincs = len(coinc_indices)

        # Store number of antennas per coincidence event
        self.nants = np.zeros(ncoincs,dtype='int')

        # Filter out coincidences with small number of antennas
        # A bit of complexity here since the number of antennas involved per coincidence event
        # will vary, so we need to keep track of the number of antennas per event.
        self.ncoincs = 0
        self.nantsmax = 0
        for index in coinc_indices:
            current_mask = (coinc_index_array==index)
            current_length = np.sum(current_mask)
            if current_length>3:
                self.nants[self.ncoincs] = current_length
                self.nantsmax = np.maximum(self.nantsmax, current_length)
                self.ncoincs += 1

        print(self.nants,self.ncoincs)
        # Now create the structure and populate it
        self.antenna_index_array = np.zeros((self.ncoincs,self.nantsmax),dtype='int')
        self.antenna_coords_array= np.zeros((self.ncoincs,self.nantsmax,3))
        self.coinc_index_array   = np.zeros((self.ncoincs,self.nantsmax),dtype='int')
        self.peak_time_array     = np.zeros((self.ncoincs,self.nantsmax))
        self.peak_amp_array      = np.zeros((self.ncoincs,self.nantsmax))

        # Filter and read
        current_coinc = 0
        for index in coinc_indices:
            mask = (coinc_index_array==index)
            print (mask)
            current_length = np.sum(mask)
            if current_length>3:
                # Next line assumes that the antenna coordinate files gives all antennas in order, starting from antenna number=init_ant
                # This will be needed to get antenna coordinates per coincidence event, from the full list in antenna_set
                self.antenna_index_array[current_coinc,:self.nants[current_coinc]] = antenna_index_array[mask]-self.ant_set.init_ant
                self.antenna_coords_array[current_coinc,:self.nants[current_coinc],:] = self.ant_set.coordinates[self.antenna_index_array[current_coinc,:self.nants[current_coinc]]]
                # Now read coincidence index (constant within the same coincidence event !), peak time and peak amplitudes per involved antennas.
                self.coinc_index_array[current_coinc,:self.nants[current_coinc]] = coinc_index_array[mask]
                self.peak_time_array[current_coinc,:self.nants[current_coinc]] = peak_time_array[mask]
                self.peak_time_array[current_coinc,:self.nants[current_coinc]] -= np.min(self.peak_time_array[current_coinc,:self.nants[current_coinc]])
                self.peak_amp_array[current_coinc,:self.nants[current_coinc]] = peak_amp_array[mask]
                current_coinc += 1
        return

class setup:

    def __init__(self, data_dir,recons_type):

        self.recons_type = recons_type
        self.data_dir    = data_dir
        if (self.recons_type<0 or self.recons_type>2):
            print("Choose reconstruction type values in :")
            print("0: plane wave reconstruction")
            print("1: spherical wave reconstruction")
            print("2: ADF model reconstruction")
            print("Other values not supported.")
            return
        if (not os.path.exists(self.data_dir)):
            print("Data directory %s does not seem to exist."%self.data_dir)
            return
        # Prepare output files
        if (self.recons_type==0):
            self.outfile = self.data_dir+'/Rec_plane_wave_recons.txt'
        elif (self.recons_type==1):
            self.outfile = self.data_dir+'/Rec_sphere_wave_recons.txt'
        elif (self.recons_type==2):
            self.outfile = self.data_dir+'/Rec_adf_recons.txt'

        # Prepare input files, depending on reconstruction type
        if (self.recons_type==1 or self.recons_type==2):
            self.input_angles_file = self.data_dir+'/Rec_plane_wave_recons.txt'
        if (self.recons_type==2):
            self.input_xmax_file = self.data_dir+'/Rec_sphere_wave_recons.txt'

    def read_angles(self,read_coinc_index=True):

        # Read theta,phi results of plane wave fit. Only makes sense if recons=0 has been run first.
        fid = open(self.input_angles_file,'r')
        l = fid.readline().strip().split()
        self.theta_rec = float(l[2])
        self.phi_rec   = float(l[4])
        if (read_coinc_index):
            self.coinc_index_rec_plane = int(l[0])
        fid.close()
        return

    def read_xmax(self,read_coinc_index=True):

        # Read xmax,ymax,zmax results of spherical wave fit. Only makes sense if recons=1 has been run first.
        fid = open(self.input_xmax_file)
        l = fid.readline().strip().split()
        self.xmax = float(l[3])
        self.ymax = float(l[4])
        self.zmax = float(l[5])
        if (read_coinc_index):
            self.coinc_index_rec_sphere = int(l[0])
        fid.close()
        return

def main():

    if (len(sys.argv) != 3):
        print ("Usage: python recons.py <recons_type> <data_dir> ")
        print ("recons_type = 0 (plane wave), 1 (spherical wave), 2 (ADF)")
        print ("data_dir is the directory containing the coincidence files")
        sys.exit(1)

    recons_type = int(sys.argv[1])
    data_dir = sys.argv[2]

    print('recons_type = ',recons_type)
    print('data_dir = ',data_dir)

    # Read antennas indices and coordinates
    an = antenna_set(data_dir+'/coord_antennas.txt')
    # Read coincidences (antenna index, coincidence index, peak time, peak amplitude)
    # Routine only keep valid number of antennas (>3)
    co = coincidence_set(data_dir+'/Rec_coinctable.txt',an)

    # Initialize reconstruction
    st = setup(data_dir,recons_type)

    if (st.recons_type==0):
        for current_recons in range(co.ncoincs):
            params_in = [np.pi,np.pi]
            res = so.minimize(PWF_loss,params_in,args=(co.antenna_coords_array[current_recons,:],co.peak_time_array[current_recons,:]),
                method='L-BFGS-B', bounds=[[0.,np.pi],[0.,2*np.pi]])
            print(res)



    return

if __name__ == '__main__':

    main()

