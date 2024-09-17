import numpy as np
import wavefronts_pwf as pwf
import wavefronts_swf as swf
import json
from scipy import interpolate as interp
import os
import matplotlib.pyplot as plt

import utils as utils

sigma_timing = 5e-9 # in sec
threshold = 75  # uV/m



####   The line below load an events from the DC2 library
arr = np.load('./PWF_SWF_data/9596.npy')
with open('./PWF_SWF_data/9596.json', 'r') as f:
    event_params = json.load(f)

ev_du_pos = arr[:, 1:4]
tmax_3d = arr[:, 4:7] * 1e-9 + np.random.randn(*(arr[:, 4:7].shape)) * sigma_timing
Emax_3d = arr[:, 7:10]

ev_du_ids = arr[:, 0]
n_du = len(ev_du_ids)

theta_gt = np.float32(event_params['zenith'])
phi_gt = np.float32(event_params['azimuth'])

## trouver un pt d'emission sur la bonne direction
shc_x = np.float32(event_params['shower_core_x'])
shc_y = np.float32(event_params['shower_core_y'])
shc_z = np.float32(event_params['shower_core_z'])

xmax_x = np.float32(event_params['xmax_pos_x'])
xmax_y = np.float32(event_params['xmax_pos_y'])
xmax_z = np.float32(event_params['xmax_pos_z'])


# corrections de Marion
core_alt = 1264
xmax_x += shc_x
xmax_y += shc_y
shc_z += core_alt

ll = list(np.vstack([np.arange(n_du), np.argmax(Emax_3d, axis=1)]).T)
ll = [tuple(l_) for l_ in ll]
tmax = np.array([tmax_3d[l_] for l_ in ll])
Emax = np.array([Emax_3d[l_] for l_ in ll])


id_above_threshold = np.where(Emax > threshold)[0]
n_above_threshold = len(id_above_threshold)

x_ants = ev_du_pos[id_above_threshold]

x_ants[:, 2] += core_alt
t_ants = tmax[id_above_threshold]
########

theta_pwf_rad, phi_pwf_rad = pwf.PWF_minimize_alternate_loss(x_ants, t_ants, nr=1.000136) 


## SWF ## 
# load the xmax vs zenith data
path_ = './PWF_SWF_data/'
if os.path.isfile(os.path.join(path_, 'zen_bin_centers.npy')):
    zen_bins_centers = np.load(os.path.join(path_, 'zen_bin_centers.npy'))
    binned_xmax_z = np.load(os.path.join(path_, 'binned_xmax_z_75_4.npy'))
    binned_dmax = np.load(os.path.join(path_, 'binned_dmax_75_4.npy'))

interp_xmax_z = interp.interp1d(zen_bins_centers, binned_xmax_z)
interp_dmax = interp.interp1d(zen_bins_centers, binned_dmax)

xmaxz = interp_xmax_z(theta_pwf_rad*pwf.phys_params.R2D)
dmax = interp_dmax(theta_pwf_rad*pwf.phys_params.R2D)

xeff = np.cos(phi_pwf_rad) * dmax
yeff = np.sin(phi_pwf_rad) * dmax
zeff = xmaxz

initial_guess = np.array([xeff, yeff, zeff, t_ants.min()])

swf_fit = swf.get_SWF_fit(x_ants, t_ants, initial_guess, sigma_t=sigma_timing, ncall=200)

K_recons_swf = np.array([-swf_fit[0]+shc_x, -swf_fit[1]+shc_y, -swf_fit[2]+shc_z])
K_recons_swf /= np.linalg.norm(K_recons_swf)

tt, pp = utils.k_to_theta_phi(K_recons_swf)

print('theta_gt = {}, theta_pwf = {}, theta_swf = {}'.format(theta_gt, theta_pwf_rad*180/np.pi, tt * 180/np.pi))
print('phi_gt = {}, phi_pwf = {}, phi_swf = {}'.format(phi_gt, phi_pwf_rad*180/np.pi, pp * 180/np.pi))
