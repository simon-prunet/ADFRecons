import numpy as np

from simulation import Simulations
s = Simulations('/Users/mguelfan/Documents/GRAND/ADF_DC2/ADFRecons/Chiche/coord_antennas.txt', '/Users/mguelfan/Documents/GRAND/ADF_DC2/output_recons_ADF/Rec_coinctable_simulated_PWF.txt')
params = np.deg2rad(np.array([180,0.])) # Theta, Phi
s(params, sigma_t = 0, simulation_type='PWF')