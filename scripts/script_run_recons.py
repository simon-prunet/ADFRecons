import os
import sys
import glob
import re
import mod_recons_tools as recons
import numpy as np
import pandas as pd
from sradio.io.shower.zhaires_master import ZhairesMaster
################################################################################
#Exemple script for reconstruction procedure
################################################################################
#Paths

#file_coord = '/Users/mguelfan/Documents/GRAND/ADF_DC2/output_recons_linear/coord_antennas.txt'
#file_input_simu = '/Users/mguelfan/Documents/GRAND/ADF_DC2/output_recons_linear/input_simus.txt'
file_recons = '/Users/mguelfan/Documents/GRAND/ADF_DC2/ADFRecons/PTREND/recons.py'
#output_directory = '/Users/mguelfan/Documents/GRAND/ADF_DC2/output_recons_linear/'
output_simu = '/Users/mguelfan/Documents/GRAND/ADF_DC2/simus/'
output_directory = '/Users/mguelfan/Documents/GRAND/ADF_DC2/output_recons_test/aggressive/'
output_directory_aggressive = '/Users/mguelfan/Documents/GRAND/ADF_DC2/output_recons_test/aggressive/'
output_directory_conservative = '/Users/mguelfan/Documents/GRAND/ADF_DC2/output_recons_test/conservative/'
#output_directory = '/Users/mguelfan/Documents/GRAND/ADF_DC2/output_recons_linear/conservative/'
#file_input_simu = '/sps/grand/mguelfand/DC2/output_recons_ADF/input_simus.txt'
#file_recons = '/sps/grand/mguelfand/DC2/ADFRecons/PTREND/recons.py'
#output_directory = '/sps/grand/mguelfand/DC2/output_recons_ADF/'
#output_simu = '/sps/grand/tueros/DiscreteStshpLibraryDunhuang2022/StshpOutbox/'
#############################
#############################       


def EventNumber(input_directory):
    last_numbers = re.search(r'\d+/$', input_directory)
    if last_numbers:
        eventnumber = last_numbers.group(0)
        eventnumber = [eventnumber.rstrip('/')]
    d_event = ZhairesMaster(input_directory)
    shower_parameters = d_event.get_simu_info()
    #print(shower_parameters)
    if shower_parameters['primary'] == 'Proton':
        shower_parameters['primary'] = 1
    #elif shower_parameters['primary'] == 'Iron':
    elif re.match(r'^Fe.*', shower_parameters['primary']):
        shower_parameters['primary'] = 2
    else:
        shower_parameters['primary'] = 0
    #print(eventnumber[0])
    event_id = float(str(int(shower_parameters['primary']))+str(int(shower_parameters['shower_zenith']*10))+str(int(shower_parameters['shower_azimuth']))+str(int(shower_parameters['energy']['value']*100))+str(int(eventnumber[0])))
    return event_id

def GetSimulationReconstructionParameters(input_directory, antennathreshold, amplitudethreshold, noise=False, offset=None):
    d_event = ZhairesMaster(input_directory)
    event = d_event.get_object_3dtraces()
    event.traces = event.get_traces_passband([50,200])
    #idx = event.remove_traces_low_signal(amplitudethreshold)
    peaktime= event.get_tmax_vmax()[0]*1e-9 #convert from ns to s
    peakamplitude = event.get_tmax_vmax()[1]
    if noise == True:
        peaktime += [np.random.normal(0.0, 5)*1.e-9 for i in range(len(peaktime))]
        peakamplitude += [np.random.normal(0.0, offset)/100 for i in range(len(peakamplitude))]*peakamplitude
    idx = np.where(peakamplitude >= amplitudethreshold) 
    idx = np.concatenate(idx)
    peaktime = peaktime[idx] ; peakamplitude = peakamplitude[idx]
    antennas = d_event.f_zhaires
    antennas.read_antpos_file()
    antennas.ants['idx'] = antennas.ants['idx']-1
    index =np.isin(antennas.ants['idx'], idx)
    event_id = [EventNumber(input_directory)]
    if len(peaktime) > antennathreshold:
        antenna_x = antennas.ants[index]['x']
        antenna_y = antennas.ants[index]['y']
        antenna_z = antennas.ants[index]['z']
        event_id = event_id*len(antenna_x)
        x=2
        #print(event_id)
        antenna_parameters = np.array([idx, peaktime, peakamplitude, antenna_x, antenna_y, antenna_z, event_id]).T
        shower_parameters = d_event.get_simu_info()
        if shower_parameters['primary'] == 'Proton':
            shower_parameters['primary'] = 1
        #elif shower_parameters['primary'] == 'Iron':
        elif re.match(r'^Fe.*', shower_parameters['primary']):
            shower_parameters['primary'] = 2
        else:
            shower_parameters['primary'] = 0
        if shower_parameters['energy']['unit'] == 'EeV':
            shower_parameters['energy']['unit'] = 1
        elif shower_parameters['energy']['unit'] == 'PeV':
            shower_parameters['energy']['unit'] = 2
        else:
            shower_parameters['energy']['unit'] = 0
        event_id = EventNumber(input_directory)
        input_simu = np.array([event_id, shower_parameters['shower_zenith'], shower_parameters['shower_azimuth'], shower_parameters['energy']['value'], shower_parameters['primary'], shower_parameters['x_max']['dist']*1e3, shower_parameters['sl_depth_of_max']['mean'], shower_parameters['x_max']['x']*1e3, shower_parameters['x_max']['y']*1e3, shower_parameters['x_max']['z']*1e3, len(idx), shower_parameters['energy']['unit']]).T
    else:
        print("Not enough antennas above threshold for issuing a T2 trigger")
        antenna_parameters = -1
        shower_parameters = -1
        input_simu = -1
        x=-1
    return antenna_parameters, shower_parameters, x, input_simu

def WriteAntennaPositionTable(filename, directory, antenna_params):
    fake_idx = np.arange(0, len(antenna_params[:,0]))
    antenna_x = antenna_params[:,3]
    antenna_y = antenna_params[:,4]
    antenna_z = antenna_params[:,5]
    file_path = f"{directory}{filename}"
    with open(file_path, 'w') as file:
        for i in range(len(fake_idx)):
            file.write(f"{fake_idx[i]} {antenna_x[i]} {antenna_y[i]} {antenna_z[i]}\n")
    return 0

def WriteReconsTable(filename, directory, antenna_params):
    fake_idx = np.arange(0, len(antenna_params[:,0]))
    peaktime = antenna_params[:,1]
    peakamplitude = antenna_params[:,2]
    event_number = antenna_params[:,6]
    file_path = f"{directory}{filename}"
    with open(file_path, 'w') as file:
        for i in range(len(fake_idx)):
            file.write(f"{fake_idx[i]} {event_number[i]} {peaktime[i]} {peakamplitude[i]}\n")
    return 0
'''       
def read_dictionnary(input_directory, amplitudethreshold):
    event_id = EventNumber(input_directory)
    d_event = ZhairesMaster(input_directory)
    event = d_event.get_object_3dtraces()
    event.traces = event.get_traces_passband([50,200])
    idx = event.remove_traces_low_signal(amplitudethreshold)
    shower_parameters = d_event.get_simu_info()
    if shower_parameters['primary'] == 'Proton':
        shower_parameters['primary'] = 1
    #elif shower_parameters['primary'] == 'Iron':
    elif re.match(r'^Fe.*', shower_parameters['primary']):
        shower_parameters['primary'] = 2
    else:
        shower_parameters['primary'] = 0
    if shower_parameters['energy']['unit'] == 'EeV':
        shower_parameters['energy']['unit'] = 1
    elif shower_parameters['energy']['unit'] == 'PeV':
        shower_parameters['energy']['unit'] = 2
    else:
        shower_parameters['energy']['unit'] = 0
    input_simu = np.array([event_id, shower_parameters['shower_zenith'], shower_parameters['shower_azimuth'], shower_parameters['energy']['value'], shower_parameters['primary'], shower_parameters['x_max']['dist']*1e3, shower_parameters['sl_depth_of_max']['mean'], shower_parameters['x_max']['x']*1e3, shower_parameters['x_max']['y']*1e3, shower_parameters['x_max']['z']*1e3, len(idx), shower_parameters['energy']['unit']]).T
    return input_simu
'''
def WriteInputSimu(filename, directory, shower_input_all):
    fake_idx = np.arange(0, len(shower_input_all[:,0]))
    event_id = shower_input_all[:,0]
    zenith = shower_input_all[:,1]
    azimuth = shower_input_all[:,2]
    energy = shower_input_all[:,3]
    energy_unit = shower_input_all[:,11]
    primary = shower_input_all[:,4]
    xmax_dist = shower_input_all[:,5]
    xmax_slant = shower_input_all[:,6]
    xmax_x = shower_input_all[:,7]
    xmax_y = shower_input_all[:,8]
    xmax_z = shower_input_all[:,9]
    number_antennas = shower_input_all[:,10]
    file_path = f"{directory}{filename}"
    with open(file_path, 'w') as file:
        for i in range(len(fake_idx)):
            file.write(f" {event_id[i]} {zenith[i]} {azimuth[i]} {energy[i]} {primary[i]} {xmax_dist[i]} {xmax_slant[i]} {xmax_x[i]} {xmax_y[i]} {xmax_z[i]} {number_antennas[i]} {energy_unit[i]} \n")
    with open(file_path, 'r') as file:
        lines = file.readlines()
    lines_sorted = sorted (lines, key = lambda lines: float(lines.split()[0]))
    with open(file_path, 'w') as file_sorted:
        file_sorted.writelines(lines_sorted)
    return 0       






#read_dictionnary(input_directory, file_path, file_coord)

#sample_files = random.sample(glob.glob(output_simu+'*'), 1)
'''
filename_position = "coord_antennas.txt"
filename_arrivaltime = 'Rec_coinctable.txt'
file_input_simu = 'input_simus.txt'
file_position = f'{output_directory}{filename_position}'
file_arrivaltime = f'{output_directory}{filename_arrivaltime}'
antennathreshold = 5
amplitudethreshold = 110 
antenna_params_all = []
antenna_params_all_conservative = []
antenna_params_all_aggressive = []
event_all = []
shower_input_all =[]
shower_input_all_aggressive =[]
shower_input_all_conservative =[]

for simus in glob.glob(output_simu+'*'):
    print('bonjour')
#for simus in sample_files:
    try:
        hdf5_files = [file for file in os.listdir(simus) if file.endswith('.hdf5')]
        if hdf5_files and GetSimulationReconstructionParameters(simus+'/', antennathreshold, amplitudethreshold)[2] != -1:
            #theta_simu = read_dictionnary(simus+'/', amplitudethreshold)[1]
            theta_simu = GetSimulationReconstructionParameters(simus+'/', antennathreshold, amplitudethreshold)[3][1]
            print(theta_simu)
            #print(read_dictionnary(simus+'/', amplitudethreshold)[3])
            if theta_simu > 50:
                #print(simus)
                parameters = GetSimulationReconstructionParameters(simus+'/', antennathreshold, amplitudethreshold)[0]
                parameters_aggressive = GetSimulationReconstructionParameters(simus+'/', antennathreshold, amplitudethreshold, noise=True, offset=10)[0]
                parameters_conservative = GetSimulationReconstructionParameters(simus+'/', antennathreshold, amplitudethreshold, noise=True, offset=20)[0]
                antenna_params_all.append(parameters)
                antenna_params_all_aggressive.append(parameters_aggressive)
                antenna_params_all_conservative.append(parameters_conservative)
                shower_input = GetSimulationReconstructionParameters(simus+'/', antennathreshold, amplitudethreshold)[3]
                shower_input_aggressive = GetSimulationReconstructionParameters(simus+'/', antennathreshold, amplitudethreshold, noise=True, offset=10)[3]
                shower_input_conservative = GetSimulationReconstructionParameters(simus+'/', antennathreshold, amplitudethreshold, noise=True, offset=20)[3]
                #shower_input = read_dictionnary(simus+'/', amplitudethreshold)
                shower_input_all.append(shower_input)
                shower_input_all_aggressive.append(shower_input_aggressive)
                shower_input_all_conservative.append(shower_input_conservative)
                #print(shower_input_all)
                #print(antenna_params_all[0][2,0])
            else:
                continue
        else:
            continue
    except:
        continue

antenna_params_all = np.vstack(antenna_params_all)
shower_input_all = np.vstack(shower_input_all)
antenna_params_all_aggressive = np.vstack(antenna_params_all_aggressive)
shower_input_all_aggressive = np.vstack(shower_input_all_aggressive)
antenna_params_all_conservative = np.vstack(antenna_params_all_conservative)
shower_input_all_conservative = np.vstack(shower_input_all_conservative)

WriteAntennaPositionTable(filename_position, output_directory, antenna_params_all) 
WriteReconsTable(filename_arrivaltime, output_directory, antenna_params_all)
WriteInputSimu(file_input_simu, output_directory, shower_input_all)

WriteAntennaPositionTable(filename_position, output_directory_aggressive, antenna_params_all_aggressive) 
WriteReconsTable(filename_arrivaltime, output_directory_aggressive, antenna_params_all_aggressive)
WriteInputSimu(file_input_simu, output_directory_aggressive, shower_input_all_aggressive)

WriteAntennaPositionTable(filename_position, output_directory_conservative, antenna_params_all_conservative) 
WriteReconsTable(filename_arrivaltime, output_directory_conservative, antenna_params_all_conservative)
WriteInputSimu(file_input_simu, output_directory_conservative, shower_input_all_conservative)
'''
#sys.exit()

os.system('python3 ' + file_recons + ' 0 ' + output_directory)
#sys.exit()


#############################
#ANALYSIS
#############################

# 0) Load shower parameters

# 1) Plane Wave Front Analysis

tab_plane = pd.read_csv(f'{output_directory}Rec_plane_wave_recons.txt', sep = '\s+', names=["IDsRec", "AntennaNumber", "ZenithRec", "_", "AzimuthRec", "nanan", "Chi2", "nanana", "time"])
tab_plane['ZenithRec'].fillna(-1, inplace=True)
tab_plane['AzimuthRec'].fillna(-1, inplace=True)
tab_plane['Chi2'].fillna(-1, inplace=True)
tab_plane['time'].fillna(-1, inplace=True)

#tab_plane.to_csv(f'{output_directory}Rec_plane_wave_recons.txt', sep = ' ', index = False, header = False, na_rep='NaN')

# 2) Spherical Analysis

os.system('python3 ' + file_recons + ' 1 ' + output_directory)


# 3) LDF Analysis

os.system('python3 ' + file_recons + ' 2 ' + output_directory)

tab_adf_rec = pd.read_csv(f'{output_directory}Rec_adf_recons.txt', sep = '\s+', names=["IDsRec", "nants", "ZenithRec", "nan", "AzimuthRec", "nanan", "Chi2", "nananan", "WidthRec", "AmpRec", "adf_time", "amplitude_rec"])
tab_input = pd.read_csv(f'{output_directory}input_simus_bis.txt', sep='\s+', names=["EventName_bis", "Zenith_bis", "Azimuth_bis", "Energy_bis", "Primary_bis", "XmaxDistance_bis", "SlantXmax_bis", "x_Xmax_bis", "y_Xmax_bis", "z_Xmax_bis", "AntennasNumber_bis", "energy_unit"])
indices = tab_adf_rec.index[tab_adf_rec['nants'] == -1].tolist()
#print(indices)

tab_input.loc[indices, 'Zenith_bis'] = -1

tab_input.to_csv(f'{output_directory}input_simus_adf.txt', sep = ' ', index = False, header = False, na_rep='NaN')

