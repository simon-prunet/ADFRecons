
import mod_recons_tools as recons
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

#output_directory = '/Users/mguelfan/Documents/GRAND/ADF_DC2/output_recons_MCMC/'
output_directory = '/sps/grand/mguelfand/DC2/output_recons_MCMC/'

# 1) Plane Wave Front Analysis
tab_plane = pd.read_csv(f'{output_directory}Rec_plane_wave_recons.txt', sep = '\s+', names=["IDsRec", "AntennaNumber", "ZenithRec", "_", "AzimuthRec", "nanan", "Chi2", "nanana", "time"])
tab_input = pd.read_csv(f'{output_directory}input_simus.txt', sep='\s+', names=["EventName", "Zenith", "Azimuth", "Energy", "Primary", "XmaxDistance", "SlantXmax", "x_Xmax", "y_Xmax", "z_Xmax", "AntennasNumber", "energy_unit"])

indices = tab_plane.index[tab_plane['ZenithRec'] == -1].tolist()
tab_input.loc[indices, 'Zenith'] = -1

tab_input_analysis = tab_input[tab_input["Zenith"] != -1]
tab_plane_analysis = tab_plane[tab_plane["ZenithRec"] != -1]

if np.isscalar(tab_plane_analysis['IDsRec']) :
    if tab_plane_analysis['AzimuthRec']  >= 180. : (tab_plane_analysis['AzimuthRec'] + 180) - 360                                                               #Reduce angles to 0-180°
else :
    tab_plane_analysis.loc[tab_plane_analysis['AzimuthRec'] > 350, 'AzimuthRec'] = 360 - tab_plane_analysis['AzimuthRec']

tab_input_analysis['Zenith'] = 180 - tab_input_analysis['Zenith']
tab_input_analysis['Azimuth'] = (180 + tab_input_analysis['Azimuth']) % 360

tab_input_analysis.to_csv(f'{output_directory}input_simus_planeanalysis.txt', sep = ' ', index = False, header = False, na_rep='NaN')
tab_plane_analysis.to_csv(f'{output_directory}Rec_plane_wave_recons_planeanalysis.txt', sep = ' ', index = False, header = False, na_rep='NaN')

IDsRec, AntennaNumber, ZenithRec, _, AzimuthRec, _, Chi2, _, time = np.loadtxt(f'{output_directory}Rec_plane_wave_recons_planeanalysis.txt').T
EventName, Zenith, Azimuth, Energy, Primary, XmaxDistance, SlantXmax, x_Xmax, y_Xmax, z_Xmax, AntennasNumber, energy_unit = np.loadtxt(f'{output_directory}input_simus_planeanalysis.txt').T

AzimErrors, ZenErrors = recons.ComputeAngularErrors(AzimuthRec, ZenithRec, Azimuth, Zenith)                 #Compute angles errors
AngularDistances = recons.ComputeAngularDistance(AzimuthRec, ZenithRec, Azimuth, Zenith)                    #Compute angular errors projected on sphere

print()
print("****Plane Wave Reconstruction****")
print("Mean angular error = ", np.mean(AngularDistances), " STD = ", np.std(AngularDistances))
PlaneRecStats = np.array([EventName, Azimuth, Zenith, Energy, Primary, XmaxDistance, SlantXmax, x_Xmax, y_Xmax, z_Xmax , AntennasNumber, ZenithRec, AzimuthRec, Chi2, AzimErrors, ZenErrors, AngularDistances, time]).T

e = recons.WriteToTxt(f"{output_directory}plane_wave_recons_stats.txt", PlaneRecStats)

tab_plane_recons_stats = pd.read_csv(f"{output_directory}plane_wave_recons_stats.txt", sep = '\s+',names=['EventName', 'Azimuth', 'Zenith', 'Energy', 'Primary', 'XmaxDistance', 'SlantXmax', 'x_Xmax', 'y_Xmax', 'z_Xmax', 'AntennasNumber', 'ZenithRec', 'AzimuthRec', 'Chi2', 'AzimErrors', 'ZenErrors', 'AngularDistances', 'time'])

#result = tab_plane_recons_stats.groupby('Zenith')['AngularDistances'].mean()
result = tab_plane_recons_stats.groupby('Zenith')['AngularDistances'].mean().reset_index(name='Averaged_angular_distance')
std_plane = tab_plane_recons_stats.groupby('Zenith')['AngularDistances'].std().reset_index(name='std_angular_distance')

result.to_csv('average.csv', index = False)
std_plane.to_csv('std.csv', index = False)

print('!!!!!!', std_plane)
#sys.exit()
#.columns = ['Zenith']
fig = plt.figure()
plt.hist(time, bins=100)
plt.xlabel('time [s]')
plt.title('Plane wave reconstruction: No noise. L-BFGS-B minimization')
plt.show()
fig = plt.figure()
plt.hist(AngularDistances, bins=100, range=[0,1])
plt.xlabel('Angular distance [°]')
plt.title('Plane wave reconstruction: No noise. L-BFGS-B minimization')
#plt.xlim(0,1)
plt.show()
#fig = plt.figure()
#fig = plt.scatter(result['Zenith'], result['Averaged_angular_distance'])
#plt.show()

# 2) Spherical Analysis
tab_sphere_rec = pd.read_csv(f'{output_directory}Rec_sphere_wave_recons.txt', sep = '\s+', names=["IDsRec", "nants", "Chi2", "_", "XSourceRec", "YSourceRec", "ZSourceRec", "TSourceRec", "time_spherical_recons"])
tab_sphere_rec_analysis = tab_sphere_rec[tab_sphere_rec["nants"] != -1]

tab_input = pd.read_csv(f'{output_directory}input_simus_bis.txt', sep='\s+', names=["EventName_bis", "Zenith_bis", "Azimuth_bis", "Energy_bis", "Primary_bis", "XmaxDistance_bis", "SlantXmax_bis", "x_Xmax_bis", "y_Xmax_bis", "z_Xmax_bis", "AntennasNumber_bis", "energy_unit"])
tab_input_analysis = tab_input[tab_input["Zenith_bis"] != -1]
#print(tab_sphere_rec)

tab_plane_rec = pd.read_csv(f'{output_directory}Rec_plane_wave_reconsbis.txt', sep = '\s+', names=['IDsRec_bis', '_', 'ZenithRec_bis', '_nan', 'AzimuthRec_bis','__', 'Chi2_bis', '_nan_', 'time_plane_recons'])
tab_plane_rec_analysis = tab_plane_rec[tab_plane_rec["_"] != -1]

if np.isscalar(tab_plane_rec_analysis['IDsRec_bis']) :
    if tab_plane_rec_analysis['AzimuthRec_bis']  >= 180. : (tab_plane_rec_analysis['AzimuthRec_bis'] + 180) - 360                                                               #Reduce angles to 0-180°
else :
    tab_plane_rec_analysis.loc[tab_plane_rec_analysis['AzimuthRec_bis'] > 350, 'AzimuthRec_bis'] = 360 - tab_plane_rec_analysis['AzimuthRec_bis']

tab_input_analysis['Zenith_bis'] = 180 - tab_input_analysis['Zenith_bis']
tab_input_analysis['Azimuth_bis'] = (180 + tab_input_analysis['Azimuth_bis']) % 360
#print(tab_input['Azimuth_bis'])
           
if np.isscalar(tab_input_analysis['Zenith_bis']) :
    InjectionHeight = 1.e5                                                  #upper atmosphere 100km
    ShowerCoreHeight = 1086.#2900.                                                #depends on simulation...
else:
    InjectionHeight = np.repeat(1.e5, len(tab_input_analysis['Zenith_bis']))
    ShowerCoreHeight = np.repeat(1086., len(tab_input_analysis['Zenith_bis']))#np.repeat(2900., len(Zenith))

  
XError = tab_input_analysis['x_Xmax_bis'] - tab_sphere_rec_analysis['XSourceRec']
YError = tab_input_analysis['y_Xmax_bis'] - tab_sphere_rec_analysis["YSourceRec"]
ZError = tab_input_analysis['z_Xmax_bis'] - tab_sphere_rec_analysis['ZSourceRec']

print()
print("****Spherical Wave Reconstruction****")
print("Mean X error = ", np.mean(XError), " STD = ", np.std(XError))
print("Mean Y error = ", np.mean(YError), " STD = ", np.std(YError))
print("Mean Z error = ", np.mean(ZError), " STD = ", np.std(ZError))

#print(tab_plane_rec['ZenithRec_bis'])

tab_input_analysis.to_csv(f'{output_directory}input_simus_bis_sphereanalysis.txt', sep = ' ', index = False, header = False, na_rep='NaN')
tab_plane_rec_analysis.to_csv(f'{output_directory}Rec_plane_wave_recons_sphereanalysis.txt', sep = ' ', index = False, header = False, na_rep='NaN')
tab_sphere_rec_analysis.to_csv(f'{output_directory}Rec_sphere_wave_recons_sphereanalysis.txt', sep = ' ', index = False, header = False, na_rep='NaN')
IDsRec, nants, Chi2, _, XSourceRec, YSourceRec, ZSourceRec, TSourceRec, time_spherical_recons = np.loadtxt(f'{output_directory}Rec_sphere_wave_recons_sphereanalysis.txt').T
EventName_bis, Zenith_bis, Azimuth_bis, Energy_bis, Primary_bis, XmaxDistance_bis, SlantXmax_bis, x_Xmax_bis, y_Xmax_bis, z_Xmax_bis, AntennasNumber_bis, energy_unit = np.loadtxt(f'{output_directory}input_simus_bis_sphereanalysis.txt').T
IDsRec_bis, _, ZenithRec_bis, _, AzimuthRec_bis, _, Chi2_bis, _, time_plane_recons = np.loadtxt(f'{output_directory}Rec_plane_wave_recons_sphereanalysis.txt').T

#tout remettre dans un fichier texte
LongitudinalError, LateralError = recons.ComputeSourceError_Long_Lat(Azimuth_bis, Zenith_bis, x_Xmax_bis, y_Xmax_bis, z_Xmax_bis, XSourceRec, YSourceRec, ZSourceRec)

print("Mean longitudinal error = ", np.mean(LongitudinalError), " STD = ", np.std(LongitudinalError))
print("Mean lateral error = ", np.mean(LateralError), " STD = ", np.std(LateralError))

GrammageError = recons.ComputeSourceErrorGrammage(SlantXmax_bis, AzimuthRec_bis, ZenithRec_bis, XSourceRec, YSourceRec, ZSourceRec, InjectionHeight, ShowerCoreHeight)
GrammageRecons, GrammageError, LongitudinalDistance_Xmax, LongitudinalDistance_Source = recons.ComputeSourceErrorGrammage_alternative_method(Azimuth_bis, Zenith_bis, x_Xmax_bis, y_Xmax_bis, z_Xmax_bis, AzimuthRec_bis, ZenithRec_bis, XSourceRec, YSourceRec, ZSourceRec, InjectionHeight, ShowerCoreHeight, XmaxDistance_bis)

print("Mean Grammage error = ", np.mean(GrammageError), " STD = ", np.std(GrammageError))

SphereRecStats = np.array([EventName_bis, Azimuth_bis, Zenith_bis, Energy_bis, Primary_bis, XmaxDistance_bis, SlantXmax_bis, x_Xmax_bis, y_Xmax_bis, z_Xmax_bis, AntennasNumber_bis, ZenithRec_bis, AzimuthRec_bis, Chi2_bis, XSourceRec, YSourceRec, ZSourceRec, TSourceRec, GrammageRecons, XError, YError, ZError, LongitudinalError, LateralError, GrammageError, LongitudinalDistance_Xmax, LongitudinalDistance_Source, time_spherical_recons]).T
#SphereRecStats = np.array([EventName_bis, Azimuth_bis, Zenith_bis, Energy_bis, Primary_bis, XmaxDistance_bis, SlantXmax_bis, x_Xmax_bis, y_Xmax_bis, z_Xmax_bis, AntennasNumber_bis, ZenithRec_bis, AzimuthRec_bis, Chi2_bis, XSourceRec, YSourceRec, ZSourceRec, TSourceRec, XError, YError, ZError]).T
f = recons.WriteToTxt(f"{output_directory}Sphere_wave_recons_stats.txt", SphereRecStats)

fig = plt.figure()
plt.hist(time_spherical_recons, bins=100)
plt.xlabel('time [s]')
plt.title('Spherical wave wave reconstruction: No noise. Gradient descent (L-BFGS-B)')
plt.show()

fig = plt.figure()
plt.hist(LongitudinalError, bins=100)
plt.xlabel('Longitudinal error [m]')
plt.show()

fig = plt.figure()
plt.hist(LateralError, bins=100)
plt.xlabel('Lateral error[m]')
plt.show()
