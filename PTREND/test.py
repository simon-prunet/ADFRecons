#import mod_recons_tools as recons
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys
import os
#from recons import antenna_set, coincidence_set, PWF_loss


font = { 'weight' : 'normal', 'size'   : 18}
plt.rc('font', **font)
plt.rcParams['figure.figsize'] = (11,7)
plt.rc('legend',fontsize=14)
parameters = {'axes.labelsize': 18}
plt.rcParams.update(parameters)

output_directory = '/Users/mguelfan/Documents/GRAND/ADF_DC2/output_recons_simusgp300_Dunhuang2022/complete_pipeline_nonoise/'
tab_antennas = pd.read_csv(f'{output_directory}coord_antennas.txt', sep = '\s+', names=["index", "x_ground", "y_ground", "z_ground"])
tab_amplitude = pd.read_csv(f'{output_directory}Rec_coinctable.txt', sep = '\s+', names=["index", "EventName", "time", "amplitude"])

merged_tab = pd.merge(tab_antennas, tab_amplitude, on='index', suffixes=('_antennas', '_amplitude'))

x_mean_position = merged_tab.groupby('EventName')['x_ground'].mean().reset_index()
y_mean_position = merged_tab.groupby('EventName')['y_ground'].mean().reset_index()
z_mean_position = merged_tab.groupby('EventName')['z_ground'].mean().reset_index()

mean_position = pd.concat([x_mean_position, y_mean_position.drop(columns=['EventName']), z_mean_position.drop(columns=['EventName'])], axis=1)
print(mean_position.head(2))

'''
an =  antenna_set(f'{output_directory}coord_antennas.txt')
co =  coincidence_set(f'{output_directory}Rec_coinctable.txt',an)
current_recons = 1
Xcore = np.mean(co.antenna_coords_array[current_recons,:co.nants[current_recons]], axis = 0)
print(Xcore)
'''
c_light = 2.997924580e8
R_earth = 6371007.0
ns = 325
kr = -0.1218
groundAltitude = 1086.0

def RefractionIndexAtPosition(X):

    R2 = X[0]*X[0] + X[1]*X[1]
    h = (np.sqrt( (X[2]+R_earth)**2 + R2 ) - R_earth)/1e3 # Altitude in km
    rh = ns*np.exp(kr*h)
    n = 1.+1e-6*rh
    #n = 1.+(1e-6*rh)/2
    return (n)

X = np.array([18, 15, 24])

core = np.array([3066, 385, 0])
Xnew = X +core
print(RefractionIndexAtPosition(X))
print(RefractionIndexAtPosition(Xnew))