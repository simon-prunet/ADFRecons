import mod_recons_tools as recons
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

#output_directory = '/Users/mguelfan/Documents/GRAND/ADF_DC2/output_recons/output_recons_nonoise/'
output_directory = '/Users/mguelfan/Documents/GRAND/ADF_DC2/output_recons_simusgp300_Dunhuang2022/complete_pipeline_aggressive/'


# 1) Plane Wave Front Analysis

#-------------------------------------------------------------------------------------------------------------------------------------------------#           
'''
tab_plane: 
output of the plane wave reconstruction : 
-ZenithRec and AzimuthRec in GRAND coordinates in ° (values -1 when the reconstruction failes)
-execution time of the plane wave reconstruction in s

tab_input: 
input of the simulation: 
-Zenith and Azimuth in AIRES conventions in °
-Primary: 1 = proton, 2 = iron, 0 = gamma
-energy in PeV (energy_unit = 2) or EeV (energy_unit = 1)
-XmaxDistance, x_Xmax, y_Xmax, z_Xmax in meters
-SlantXmax in g/cm2
'''
#-------------------------------------------------------------------------------------------------------------------------------------------------#
tab_plane = pd.read_csv(f'{output_directory}Rec_plane_wave_recons.txt', sep = '\s+', names=["IDsRec", "AntennaNumber", "ZenithRec", "_", "AzimuthRec", "nanan", "Chi2", "nanana", "time"])
tab_input = pd.read_csv(f'{output_directory}input_simus.txt', sep='\s+', names=["EventName", "Zenith", "Azimuth", "Energy", "Primary", "XmaxDistance", "SlantXmax", "x_Xmax", "y_Xmax", "z_Xmax", "AntennasNumber", "energy_unit"])

tab_input = tab_input.sort_values(by=tab_input.columns[0])

'''remove all the lines with -1 (error or no convergence)'''
indices = tab_plane.index[tab_plane['ZenithRec'] == -1].tolist()
tab_input.loc[indices, 'Zenith'] = -1
tab_input_analysis = tab_input[tab_input["Zenith"] != -1]
tab_plane_analysis = tab_plane[tab_plane["ZenithRec"] != -1]

'''remove all the simu where no convergence with PWF for the zenith angle'''
'''only keep down-going air showers'''
bounds_zenith = [np.rad2deg(np.pi/2), np.rad2deg(np.pi)] 
indices_low = tab_plane_analysis.index[tab_plane_analysis['ZenithRec'] <= bounds_zenith[0]].tolist()
tab_input_analysis.loc[indices_low, 'Zenith'] = -1
indices_high = tab_plane_analysis.index[tab_plane_analysis['ZenithRec'] >= bounds_zenith[1]].tolist()
tab_input_analysis.loc[indices_high, 'Zenith'] = -1

tab_input_analysis = tab_input_analysis[tab_input_analysis["Zenith"] != -1]
tab_plane_analysis = tab_plane_analysis[tab_plane_analysis["ZenithRec"] > bounds_zenith[0]]
tab_plane_analysis = tab_plane_analysis[tab_plane_analysis["ZenithRec"] < bounds_zenith[1]]

#if np.isscalar(tab_plane_analysis['IDsRec']) :
#   if tab_plane_analysis['AzimuthRec']  >= 180. : (tab_plane_analysis['AzimuthRec'] + 180) - 360                                                               #Reduce angles to 0-180°
#else :
#    tab_plane_analysis.loc[tab_plane_analysis['AzimuthRec'] > 360, 'AzimuthRec'] = 360 - tab_plane_analysis['AzimuthRec']

#convert from ZhaIRES conventions to GRAND conventions 
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
PlaneRecStats = np.array([IDsRec, Azimuth, Zenith, Energy, Primary, XmaxDistance, SlantXmax, x_Xmax, y_Xmax, z_Xmax , AntennasNumber, ZenithRec, AzimuthRec, Chi2, AzimErrors, ZenErrors, AngularDistances, time]).T

e = recons.WriteToTxt(f"{output_directory}plane_wave_recons_stats.txt", PlaneRecStats)

tab_plane_recons_stats = pd.read_csv(f"{output_directory}plane_wave_recons_stats.txt", sep = '\s+',names=['EventName', 'Azimuth', 'Zenith', 'Energy', 'Primary', 'XmaxDistance', 'SlantXmax', 'x_Xmax', 'y_Xmax', 'z_Xmax', 'AntennasNumber', 'ZenithRec', 'AzimuthRec', 'Chi2', 'AzimErrors', 'ZenErrors', 'AngularDistances', 'time'])
os.remove(f'{output_directory}input_simus_planeanalysis.txt')
os.remove(f'{output_directory}Rec_plane_wave_recons_planeanalysis.txt')

#sys.exit()
#-------------------------------------------------------------------------------------------------------------------------------------------------#           
'''
tab_sphere_rec: 
output of the spherical wave reconstruction : 
-XSourceRec, YSourceRec, ZSourceRec in meters (-1 if the PWF failed)
-TSourceRec in meters 
-execution time of the sphere wave reconstruction in s

tab_input: 
input of the simulation: 
-Zenith and Azimuth in AIRES conventions in °
-Primary: 1 = proton, 2 = iron, 0 = gamma
-energy in PeV (energy_unit = 2) or EeV (energy_unit = 1)
-XmaxDistance, x_Xmax, y_Xmax, z_Xmax in meters
-SlantXmax in g/cm2
'''
#-------------------------------------------------------------------------------------------------------------------------------------------------#
# 2) Spherical Analysis
tab_sphere_rec = pd.read_csv(f'{output_directory}Rec_sphere_wave_recons.txt', sep = '\s+', names=["IDsRec", "nants", "Chi2", "_", "XSourceRec", "YSourceRec", "ZSourceRec", "TSourceRec", "rho_Xsource", "time_spherical_recons"])
tab_sphere_rec_analysis = tab_sphere_rec[tab_sphere_rec["nants"] != -1]

tab_input = pd.read_csv(f'{output_directory}input_simus_bis.txt', sep='\s+', names=["EventName", "Zenith", "Azimuth", "Energy", "Primary", "XmaxDistance", "SlantXmax", "x_Xmax", "y_Xmax", "z_Xmax", "AntennasNumber", "energy_unit"])
tab_input = tab_input.sort_values(by=tab_input.columns[0])
tab_input_analysis = tab_input[tab_input["Zenith"] != -1]


tab_plane_rec = pd.read_csv(f'{output_directory}Rec_plane_wave_reconsbis.txt', sep = '\s+', names=['IDsRec', '_', 'ZenithRec', '_nan', 'AzimuthRec','__', 'Chi2', '_nan_', 'time_plane_recons'])
tab_plane_rec_analysis = tab_plane_rec[tab_plane_rec["_"] != -1]

#if np.isscalar(tab_plane_rec_analysis['IDsRec']) :
#    if tab_plane_rec_analysis['AzimuthRec']  >= 180. : (tab_plane_rec_analysis['AzimuthRec'] + 180) - 360                                                               #Reduce angles to 0-180°
#else :
#    tab_plane_rec_analysis.loc[tab_plane_rec_analysis['AzimuthRec'] > 360, 'AzimuthRec'] = 360 - tab_plane_rec_analysis['AzimuthRec']

#convert from ZhaIRES conventions to GRAND conventions 
tab_input_analysis['Zenith'] = 180 - tab_input_analysis['Zenith']
tab_input_analysis['Azimuth'] = (180 + tab_input_analysis['Azimuth']) % 360

if np.isscalar(tab_input_analysis['Zenith']) :
    InjectionHeight = 1.e5                                                  #upper atmosphere 100km
    ShowerCoreHeight = 1086.#2900.                                                #depends on simulation...
else:
    InjectionHeight = np.repeat(1.e5, len(tab_input_analysis['Zenith']))
    ShowerCoreHeight = np.repeat(1086., len(tab_input_analysis['Zenith']))#np.repeat(2900., len(Zenith))

  
XError = tab_input_analysis['x_Xmax'] - tab_sphere_rec_analysis['XSourceRec']
YError = tab_input_analysis['y_Xmax'] - tab_sphere_rec_analysis["YSourceRec"]
ZError = tab_input_analysis['z_Xmax'] - tab_sphere_rec_analysis['ZSourceRec']


print()
print("****Spherical Wave Reconstruction****")
print("Mean X error = ", np.mean(XError), " STD = ", np.std(XError))
print("Mean Y error = ", np.mean(YError), " STD = ", np.std(YError))
print("Mean Z error = ", np.mean(ZError), " STD = ", np.std(ZError))

#print(tab_plane_rec['ZenithRec'])

tab_input_analysis.to_csv(f'{output_directory}input_simus_bis_sphereanalysis.txt', sep = ' ', index = False, header = False, na_rep='NaN')
tab_plane_rec_analysis.to_csv(f'{output_directory}Rec_plane_wave_recons_sphereanalysis.txt', sep = ' ', index = False, header = False, na_rep='NaN')
tab_sphere_rec_analysis.to_csv(f'{output_directory}Rec_sphere_wave_recons_sphereanalysis.txt', sep = ' ', index = False, header = False, na_rep='NaN')
IDsRec, nants, Chi2, _, XSourceRec, YSourceRec, ZSourceRec, TSourceRec, rho_XSource, time_spherical_recons = np.loadtxt(f'{output_directory}Rec_sphere_wave_recons_sphereanalysis.txt').T
EventName_bis, Zenith_bis, Azimuth_bis, Energy_bis, Primary_bis, XmaxDistance_bis, SlantXmax_bis, x_Xmax_bis, y_Xmax_bis, z_Xmax_bis, AntennasNumber_bis, energy_unit = np.loadtxt(f'{output_directory}input_simus_bis_sphereanalysis.txt').T
IDsRec_bis, _, ZenithRec_bis, _, AzimuthRec_bis, _, Chi2_bis, _, time_plane_recons = np.loadtxt(f'{output_directory}Rec_plane_wave_recons_sphereanalysis.txt').T
#tout remettre dans un fichier texte
LongitudinalError, LateralError = recons.ComputeSourceError_Long_Lat(Azimuth_bis, Zenith_bis, x_Xmax_bis, y_Xmax_bis, z_Xmax_bis, XSourceRec, YSourceRec, ZSourceRec)
#LongitudinalError = LongitudinalError.tolist()
#LateralError = LateralError.tolist()
#for i, value in enumerate(LateralError):
#    if value > 1000:
#        print("Indice :", i)
LongitudinalError_array = np.array(LongitudinalError)
indices_nan = np.where(np.isnan(LongitudinalError_array))[0]
#print(indices_nan)
LongitudinalError = [value for value in LongitudinalError if not np.isnan(value)] #si NaN dans le calcul
LateralError = [value for value in LateralError if not np.isnan(value)] #si NaN dans le calcul


print("Mean longitudinal error = ", np.mean(LongitudinalError), " STD = ", np.std(LongitudinalError))
print("Mean lateral error = ", np.mean(LateralError), " STD = ", np.std(LateralError))

'''
Azimuth_bis = np.delete(Azimuth_bis, indices_nan)
Zenith_bis = np.delete(Zenith_bis, indices_nan)
x_Xmax_bis = np.delete(x_Xmax_bis, indices_nan)
y_Xmax_bis = np.delete(y_Xmax_bis, indices_nan)
z_Xmax_bis = np.delete(z_Xmax_bis, indices_nan)
AzimuthRec_bis = np.delete(AzimuthRec_bis, indices_nan)
ZenithRec_bis = np.delete(ZenithRec_bis, indices_nan)
XSourceRec = np.delete(XSourceRec, indices_nan)
YSourceRec = np.delete(YSourceRec, indices_nan)
ZSourceRec = np.delete(ZSourceRec, indices_nan)
InjectionHeight = np.delete(InjectionHeight, indices_nan)
ShowerCoreHeight = np.delete(ShowerCoreHeight, indices_nan)
XmaxDistance_bis = np.delete(XmaxDistance_bis, indices_nan)
EventName_bis = np.delete(EventName_bis, indices_nan)
Energy_bis = np.delete(Energy_bis, indices_nan)
Primary_bis = np.delete(Primary_bis, indices_nan)
SlantXmax_bis = np.delete(SlantXmax_bis, indices_nan)
AntennasNumber_bis = np.delete(AntennasNumber_bis, indices_nan)
Chi2_bis = np.delete(Chi2_bis, indices_nan)
TSourceRec = np.delete(TSourceRec, indices_nan)
time_spherical_recons = np.delete(time_spherical_recons, indices_nan)
XError = np.delete(XError, indices_nan)
YError = np.delete(YError, indices_nan)
ZError = np.delete(ZError, indices_nan)
'''


#GrammageError = recons.ComputeSourceErrorGrammage(SlantXmax_bis, AzimuthRec_bis, ZenithRec_bis, XSourceRec, YSourceRec, ZSourceRec, InjectionHeight, ShowerCoreHeight)
GrammageRecons, GrammageError, LongitudinalDistance_Xmax, LongitudinalDistance_Source = recons.ComputeSourceErrorGrammage_alternative_method(Azimuth_bis, Zenith_bis, x_Xmax_bis, y_Xmax_bis, z_Xmax_bis, AzimuthRec_bis, ZenithRec_bis, XSourceRec, YSourceRec, ZSourceRec, InjectionHeight, ShowerCoreHeight, XmaxDistance_bis)

#LateralAngle = recons.ComputeLareralAngle(x_Xmax_bis, y_Xmax_bis, z_Xmax_bis, LateralError)
LateralAngle_alternatif = recons.ComputeLareralAngle_alternative_method(x_Xmax_bis, y_Xmax_bis,  z_Xmax_bis, XSourceRec, YSourceRec, ZSourceRec, ShowerCoreHeight)
#LateralAngle = recons.ComputeLareralAngle(XSourceRec, YSourceRec, ZSourceRec, LateralError)
#print(LateralAngle_alternatif)
#print(min(LongitudinalDistance_Xmax))

print("Mean Grammage error = ", np.mean(GrammageError), " STD = ", np.std(GrammageError))

SphereRecStats = np.array([EventName_bis, Azimuth_bis, Zenith_bis, Energy_bis, Primary_bis, XmaxDistance_bis, SlantXmax_bis, x_Xmax_bis, y_Xmax_bis, z_Xmax_bis, AntennasNumber_bis, ZenithRec_bis, AzimuthRec_bis, Chi2_bis, XSourceRec, YSourceRec, ZSourceRec, TSourceRec, GrammageRecons, XError, YError, ZError, LongitudinalError, LateralError, GrammageError, LongitudinalDistance_Xmax, LongitudinalDistance_Source, time_spherical_recons, LateralAngle_alternatif]).T
#SphereRecStats = np.array([EventName_bis, Azimuth_bis, Zenith_bis, Energy_bis, Primary_bis, XmaxDistance_bis, SlantXmax_bis, x_Xmax_bis, y_Xmax_bis, z_Xmax_bis, AntennasNumber_bis, ZenithRec_bis, AzimuthRec_bis, Chi2_bis, XSourceRec, YSourceRec, ZSourceRec, TSourceRec, XError, YError, ZError]).T
#SphereRecStats = np.array([EventName_bis, Azimuth_bis, Zenith_bis, Energy_bis, Primary_bis, XmaxDistance_bis, SlantXmax_bis, x_Xmax_bis, y_Xmax_bis, z_Xmax_bis, AntennasNumber_bis, ZenithRec_bis, AzimuthRec_bis, Chi2_bis, XSourceRec, YSourceRec, ZSourceRec, TSourceRec, GrammageRecons, LongitudinalError, LateralError, GrammageError, LongitudinalDistance_Xmax, LongitudinalDistance_Source, time_spherical_recons]).T
f = recons.WriteToTxt(f"{output_directory}Sphere_wave_recons_stats.txt", SphereRecStats)
os.remove(f'{output_directory}Rec_plane_wave_recons_sphereanalysis.txt')
os.remove(f'{output_directory}Rec_sphere_wave_recons_sphereanalysis.txt')
os.remove(f'{output_directory}input_simus_bis_sphereanalysis.txt')

#sys.exit()
#-------------------------------------------------------------------------------------------------------------------------------------------------#           
'''
tab_adf_rec: 
output of the spherical wave reconstruction : 
-ZenithRec and AzimuthRec in ° in GRAND conventions
-WidthRec and AmpRec : parameters of the analytical ADF
-execution time of the ADF reconstruction in s

tab_input: 
input of the simulation: 
-Zenith and Azimuth in AIRES conventions in °
-Primary: 1 = proton, 2 = iron, 0 = gamma
-energy in PeV (energy_unit = 2) or EeV (energy_unit = 1)
-XmaxDistance, x_Xmax, y_Xmax, z_Xmax in meters
-SlantXmax in g/cm2
'''
#-------------------------------------------------------------------------------------------------------------------------------------------------#
# 3) ADF Analysis
tab_adf_rec = pd.read_csv(f'{output_directory}Rec_adf_recons.txt', sep = '\s+', names=["IDsRec", "nants", "ZenithRec", "nan", "AzimuthRec", "nanan", "Chi2", "nananan", "WidthRec", "AmpRec", "adf_time"])
tab_input = pd.read_csv(f'{output_directory}input_simus_bis.txt', sep='\s+', names=["EventName_bis", "Zenith_bis", "Azimuth_bis", "Energy_bis", "Primary_bis", "XmaxDistance_bis", "SlantXmax_bis", "x_Xmax_bis", "y_Xmax_bis", "z_Xmax_bis", "AntennasNumber_bis", "energy_unit"])
tab_input = tab_input.sort_values(by=tab_input.columns[0])

tab_allsimu = pd.read_csv(f'{output_directory}input_simus.txt', sep='\s+', names=["EventName", "Zenith", "Azimuth", "Energy", "Primary", "XmaxDistance", "SlantXmax", "x_Xmax", "y_Xmax", "z_Xmax", "AntennasNumber", "energy_unit"])

indices = tab_adf_rec.index[tab_adf_rec['nants'] == -1].tolist()

tab_input.loc[indices, 'Zenith_bis'] = -1

tab_input_analysis = tab_input[tab_input["Zenith_bis"] != -1]
tab_adf_rec_analysis = tab_adf_rec[tab_adf_rec["nants"] != -1]

'''remove all the simu where no convergence with ADF for the width'''
bounds_width = [0.1, 3.0] 
indices_low = tab_adf_rec_analysis.index[tab_adf_rec_analysis['WidthRec'] == bounds_width[0]].tolist()
tab_input_analysis.loc[indices_low, 'Zenith_bis'] = -1
indices_high = tab_adf_rec_analysis.index[tab_adf_rec_analysis['WidthRec'] == bounds_width[1]].tolist()
tab_input_analysis.loc[indices_high, 'Zenith_bis'] = -1

tab_input_analysis = tab_input_analysis[tab_input_analysis["Zenith_bis"] != -1]
tab_adf_rec_analysis = tab_adf_rec_analysis[tab_adf_rec_analysis["WidthRec"] != bounds_width[0]]
tab_adf_rec_analysis = tab_adf_rec_analysis[tab_adf_rec_analysis["WidthRec"] != bounds_width[1]]

'''remove all the simu where no convergence with ADF for the amplitude'''
bounds_amplitude = [1e6,1e10] 
indices_low = tab_adf_rec_analysis.index[tab_adf_rec_analysis['AmpRec'] == bounds_amplitude[0]].tolist()
tab_input_analysis.loc[indices_low, 'Zenith_bis'] = -1
indices_high = tab_adf_rec_analysis.index[tab_adf_rec_analysis['AmpRec'] == bounds_amplitude[1]].tolist()
tab_input_analysis.loc[indices_high, 'Zenith_bis'] = -1

tab_input_analysis = tab_input_analysis[tab_input_analysis["Zenith_bis"] != -1]
tab_adf_rec_analysis = tab_adf_rec_analysis[tab_adf_rec_analysis["AmpRec"] != bounds_amplitude[0]]
tab_adf_rec_analysis = tab_adf_rec_analysis[tab_adf_rec_analysis["AmpRec"] != bounds_amplitude[1]]


print(len(tab_adf_rec_analysis)/len(tab_allsimu)*100)

if np.isscalar(tab_adf_rec_analysis['IDsRec']) :
    if AzimuthRec  > 180. : AzimuthRec -= 360                                                               #Reduce angles to 0-180°
else :
    tab_adf_rec_analysis.loc[tab_adf_rec['AzimuthRec'] > 360, 'AzimuthRec'] = 360 - tab_adf_rec_analysis['AzimuthRec']

tab_input_analysis['Zenith_bis'] = 180 - tab_input_analysis['Zenith_bis']
tab_input_analysis['Azimuth_bis'] = (180 + tab_input_analysis['Azimuth_bis']) % 360

tab_adf_rec_analysis.to_csv(f'{output_directory}Rec_adf_recons_adfanalysis.txt', sep = ' ', index = False, header = False, na_rep='NaN')
tab_input_analysis.to_csv(f'{output_directory}input_simus_adfanalysis.txt', sep = ' ', index = False, header = False, na_rep='NaN')

IDsRec, nants, ZenithRec, _, AzimuthRec, _, Chi2, _, WidthRec, AmpRec, adf_time = np.loadtxt(f"{output_directory}Rec_adf_recons_adfanalysis.txt").T
EventName_bis, Zenith_bis, Azimuth_bis, Energy_bis, Primary_bis, XmaxDistance_bis, SlantXmax_bis, x_Xmax_bis, y_Xmax_bis, z_Xmax_bis, AntennasNumber_bis, energy_unit = np.loadtxt(f'{output_directory}input_simus_adfanalysis.txt').T


AzimErrors, ZenErrors = recons.ComputeAngularErrors(AzimuthRec, ZenithRec, Azimuth_bis, Zenith_bis)                 #Compute angles errors
AngularDistances = recons.ComputeAngularDistance(AzimuthRec, ZenithRec, Azimuth_bis, Zenith_bis)                    #Compute angular errors projected on sphere

print()
print("****ADF Reconstruction****")
print("Mean angular error = ", np.mean(AngularDistances), " STD = ", np.std(AngularDistances))

CerenkovRecStats = np.array([EventName_bis, Azimuth_bis, Zenith_bis, Energy_bis, Primary_bis, XmaxDistance_bis, SlantXmax_bis, x_Xmax_bis, y_Xmax_bis, z_Xmax_bis, AntennasNumber_bis, ZenithRec, AzimuthRec, Chi2, WidthRec, AmpRec, AzimErrors, ZenErrors, AngularDistances, adf_time]).T
g = recons.WriteToTxt(f"{output_directory}adf_recons_stats.txt", CerenkovRecStats)
os.remove(f'{output_directory}Rec_adf_recons_adfanalysis.txt')
os.remove(f'{output_directory}input_simus_adfanalysis.txt')