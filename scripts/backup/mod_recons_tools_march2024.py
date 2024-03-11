import sys
import subprocess as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
#import hdf5fileinout as hdf5io
#import mod_fun as modfun

################################################################################
#Constant with a k
################################################################################
kRearth = 6370949.
################################################################################
#Functions
################################################################################


################################################################################
#Reconstruction processes

def GetAllReconstructionParameters(SimulationList_, AntennasThreshold_, AmplitudeThreshold_, flag_P2PInfo=True, Verbatim=False):
    '''
    Read several hdf5 simulation files (NoTrace) and extract relevant parameters for reconstruction
    Inputs: simulation files list (NoTrace.hdf5)
    Outputs: two numpy tables ShowerParameters: [EventName, Azimuth, Zenith, Energy, Primary, XmaxDistance, SlantXmax, SelectedAntennasNumber]
                             AntannasParameters: [AntennaIDs, x, y, z, peaktime, peakamplitude]
    Containing all simulations
    '''
    # TODO: handle errors
    count = 0
    _AntennasParametersAll, _ShowersParametersAll = [], []

    for Simulation in SimulationList_ :

        if Verbatim:
            percentage = count / len(SimulationList_) * 100
            count +=1
            sp.call('clear',shell=True)
            print(f"{percentage:.1f}% done -> processing event : "+Simulation)
        #_shower, _antennas = GetSimulationReconstructionParameters(Simulation, AntennasThreshold_, AmplitudeThreshold_, flag_P2PInfo, Verbatim) #For debug
        try:
            _shower, _antennas = GetSimulationReconstructionParameters(Simulation, AntennasThreshold_, AmplitudeThreshold_, flag_P2PInfo, Verbatim)
            if any(_shower == -1) :
                if Verbatim: print("Event :", Simulation, " not enought antennas above threshold")
            else :
                _ShowersParametersAll.append(_shower)
                _AntennasParametersAll.append(_antennas)
        except:
            if Verbatim: print("Event :", Simulation, "failed")

    if len(SimulationList_) and len(_ShowersParametersAll) > 0:
        #_ShowersParametersAll = np.concatenate(_ShowersParametersAll)
        _ShowersParametersAll = np.array(_ShowersParametersAll)
        _AntennasParametersAll = np.concatenate(_AntennasParametersAll)

    return _ShowersParametersAll, _AntennasParametersAll

def GetSimulationReconstructionParameters(InputFilename_, AntennasThreshold_, AmplitudeThreshold_, flag_P2PInfo=True, Verbatim=False):
    '''
    Read an hdf5 simulation file (NoTrace) and extract relevant parameters for reconstruction
    Inputs: simulation file (NoTrace.hdf5)
    Outputs: two numpy tables ShowerParameters: [EventName, Azimuth, Zenith, Energy, Primary, XmaxDistance, SlantXmax, SelectedAntennasNumber]
                             AntannasParameters: [AntennaIDs, x, y, z, peaktime, peakamplitude]
    '''
    # TODO: handle errors
    #Shower event access
    RunInfo = hdf5io.GetRunInfo(InputFilename_)
    NumberOfEvents = hdf5io.GetNumberOfEvents(RunInfo)
    EventNumber = NumberOfEvents-1
    EventName = hdf5io.GetEventName(RunInfo,EventNumber)
    EventInfo = hdf5io.GetEventInfo(InputFilename_,EventName)

    Zenith = hdf5io.GetEventZenith(RunInfo,EventNumber)
    Azimuth = hdf5io.GetEventAzimuth(RunInfo,EventNumber)
    Primary = hdf5io.GetEventPrimary(RunInfo,EventNumber)
    Primary = GetPrimaryClassification(Primary)
    Energy = hdf5io.GetEventEnergy(RunInfo,EventNumber)
    XmaxDistance = hdf5io.GetEventXmaxDistance(RunInfo,EventNumber)
    SlantXmax = hdf5io.GetEventSlantXmax(RunInfo,EventNumber)
    x_Xmax, y_Xmax, z_Xmax = hdf5io.GetXmaxPosition(EventInfo).data[0]
    Energy = hdf5io.GetEventEnergy(RunInfo,EventNumber)

    if Verbatim : print("Azimuth = ", Azimuth, " Zenith = ", Zenith, " Energy = ", Energy, " Primary : ", Primary, " Event Name : ", EventName)

    #Antenna info
    AntennaInfo = hdf5io.GetAntennaInfo(InputFilename_,EventName)
    _AntennasParameters = -1
    _ShowersParameters = -1
    ##
    if hdf5io.GetNumberOfAntennas(AntennaInfo) > AntennasThreshold_ :                                                          #if total antennas number < T2 trigger requirements

        IDs = hdf5io.GetAntIDFromAntennaInfo(AntennaInfo)
        AntennaIDs = np.array([int(id_.replace('CrossCheck', '')[1:]) for id_ in IDs])

        Positions = hdf5io.GetAntennaPositions(AntennaInfo)
        x = Positions[0].data
        y = Positions[1].data
        z = Positions[2].data
        if flag_P2PInfo:
            AntennaP2PInfo = hdf5io.GetAntennaP2PInfo(InputFilename_,EventName)
            peaktime, peakamplitude = AntennaP2PInfo['HilbertPeakTimeE'].data, AntennaP2PInfo['HilbertPeakE'].data
            peaktime *= 1.e-9 #get peaktime in s for reconstruction
        else : #                                                                #Go through the traces
            # peaktime, peakamplitude = hdf5io.get_peak_time_hilbert_hdf5(InputFilename_, antennamax="All",antennamin=0, usetrace="efield", DISPLAY=False)
            # peaktime *= 1.e-9 #get peaktime in s for reconstruction
            #
            ##Filter E-field
            print("Filtered signal !!")
            fmin = 50.e6
            fmax = 200.e6
            peaktime, peakamplitude = modfun.get_filtered_peakAmpTime_Hilbert(InputFilename_, EventName, AntennaInfo, fmin, fmax)

        #Antennas selection
        sel = np.where(peakamplitude >= AmplitudeThreshold_)[0]                                                              #threshold selection
        AntennaIDs = AntennaIDs[sel] ; x = x[sel] ; y = y[sel] ; z = z[sel]
        peaktime = peaktime[sel] ; peakamplitude = peakamplitude[sel]
        ##
        if len(x) > AntennasThreshold_ :                                                                                         #if total antennas number < T2 trigger requirements
            if Primary == -1 : Primary = 16   # patch to change labeling of neutrino from Matias simulation
            #EventID =  np.repeat(int((f'{Primary:.1f}' + f'{Energy:.3f}' + f'{Azimuth:.3f}' + f'{Zenith:.3f}').replace(".", "") + EventName.split("_")[-1]), len(x))   #Event ID = ??
            EventID = np.repeat(int(EventName[0:15]), len(x))
            _AntennasParameters = np.array([AntennaIDs, EventID, x, y, z, peaktime, peakamplitude]).T                                       #_AntennasParameters -> (AntennaIDs, EventID, x, y, z, peaktime, peakamplitude)
            _ShowersParameters = np.array([EventID[0], Azimuth, Zenith, Energy, Primary, XmaxDistance, SlantXmax, x_Xmax, y_Xmax, z_Xmax, len(x)]).T                #_ShowersParameters -> (EventName, Azimuth, Zenith, Energy, Primary, XmaxDistance, SlantXmax, SelectedAntennasNumber)
        else:
            if Verbatim : print("Not enough antennas above threshold for issuing a T2 trigger")

    else :
        if Verbatim : print("No antennas found in ", EventName)


    return _ShowersParameters, _AntennasParameters

def WriteAntennaPositionsTable(file_name_, antennas_params_) :
    '''
    Write the coord.txt table for the TREND reconstruction software
    Inputs: file name (coord_antennas.txt) and an antenna parameters array [AntennaIDs, x, y, z, peaktime, peakamplitude] (output of GetSimulationReconstructionParameters)
    Outputs: return 0 and create a txt file
    Note: antennas ID has to start from 0 and not repeat themself -> fake antenna ID (counter starting from 0 and up to len(total antennas))
    '''
    # TODO: handle errors
    _fake_ID = np.arange(0, len(antennas_params_[:,0]))
    _input = np.array([_fake_ID, antennas_params_[:,2], antennas_params_[:,3], antennas_params_[:,4]]) #Not clean...

    _file = open(file_name_, 'w')
    np.savetxt(_file, _input.T, fmt = ''.join(['   %i']*1 + ['   %11.5f']*3 ))
    _file.close()

    return 0

def WriteReconsTable(file_name_, antennas_params_) :
    '''
    Write the coord.txt table for the TREND reconstruction software
    Inputs: file name (Rec_coinctable.txt) and an antenna parameters array [AntennaIDs, x, y, z, peaktime, peakamplitude] (output of GetSimulationReconstructionParameters)
    Outputs: return 0 and create a txt file
    Note: antennas ID has to start from 0 and not repeat themself -> fake antenna ID (counter starting from 0 and up to len(total antennas))
    '''
    # TODO: handle errors
    _fake_ID = np.arange(0, len(antennas_params_[:,0]))
    _input = np.array([_fake_ID, antennas_params_[:,1], antennas_params_[:,5], antennas_params_[:,6]])
    #Not clean...

    _file = open(file_name_, 'w')
    np.savetxt(_file, _input.T, fmt = ''.join(['   %i']*2 + ['   %11.12f']*2))
    _file.write("{0:3d} {1:3d} {2:11.3f} {3:11.3f}\n".format(10, 0, 0, 0))
    _file.close()

    return 0

def GetPrimaryClassification(Primary_):
    '''
    Toi même tu sais
    '''
    # TODO: handle errors
    _nomenklatura = -1
    if Primary_ == 'Gamma' :
        _nomenklatura = 0
    elif Primary_ == 'Proton':
        _nomenklatura = 1
    elif Primary_ == "Fe^56" :
        _nomenklatura = 2

    return _nomenklatura

def WriteToTxt(file_name_, InputsArray_):
    '''
    Write to txt file any array
    '''
    # TODO: handle errors
    _file = open(file_name_, 'w')
    #format = "%12.5le %12.5le %12.5le %12.5le %12.5le %12.5le %12.5le %12.5le %12.5le %12.5le %12.5le %12.5le %12.5le %12.5le %12.5le %12.5le %12.5le %12.5le"
    if len(InputsArray_.shape) ==1:
        #format = "%15.5le " * InputsArray_.shape[0]  # Format pour tous les éléments sur une ligne
        np.savetxt(_file,  InputsArray_.reshape(1, -1))
        #np.savetxt(_file, InputsArray_)
        _file.close()
    else:
        format = "%20.9le" * InputsArray_.shape[1]
        #format = "%d" + "%15.5le" * (InputsArray_.shape[1]-1)
        np.savetxt(_file, InputsArray_, fmt=format)
        #np.savetxt(_file, InputsArray_)
        _file.close()

    return 0

################################################################################
#Analysis tools

def ComputeAngularErrors(rec_azim, rec_zen, azim, zen) :
    '''
    Toi même tu sais
    '''
    # TODO: handle errors
    azim_err = np.arccos(np.cos((rec_azim - azim)*np.pi/180))*180/np.pi
    zen_err = rec_zen - zen

    return np.array([azim_err, zen_err])

def ComputeAngularDistance(azim_r, zen_r, azim_s, zen_s) :
    '''
    Toi même tu sais
    '''
    # TODO: handle errors
    azim_diff = azim_r - azim_s

    return 180./np.pi * np.arccos(np.cos(zen_r*np.pi/180)*np.cos(zen_s*np.pi/180) + np.cos(azim_diff*np.pi/180) * np.sin(zen_s*np.pi/180) * np.sin(zen_r*np.pi/180))

def ComputeXmax(Azimuth_, Zenith_, XmaxDistance_, ShowerCoreHeight_):
    '''
    Toi même tu sais
    '''
    # TODO: handle errors
    k_shower = np.array([np.cos(Azimuth_*np.pi/180.)*np.sin(Zenith_*np.pi/180),np.sin(Azimuth_*np.pi/180.)*np.sin(Zenith_*np.pi/180), np.cos(Zenith_*np.pi/180)])
    _x_xmax = -k_shower[0]*XmaxDistance_ ; _y_xmax = -k_shower[1]*XmaxDistance_ ; _z_xmax = ShowerCoreHeight_ - k_shower[2]*XmaxDistance_

    return _x_xmax, _y_xmax, _z_xmax



def ComputeSourceError(Azimuth_, Zenith_, XmaxDistance_, ShowerCoreHeight_, XRec_, YRec_, ZRec_):
    '''
    Toi même tu sais
    '''
    # TODO: handle errors
    _x_xmax, _y_xmax, _z_xmax = ComputeXmax(Azimuth_, Zenith_, XmaxDistance_, ShowerCoreHeight_)
    _x_error = _x_xmax - XRec_
    _y_error = _y_xmax - YRec_
    _z_error = _z_xmax - ZRec_

    return _x_error, _y_error, _z_error

def ComputeGrammage(Zenith_, XmaxDistance_, ShowerCoreHeight_, InjectionHeight_, LongitudinalDistance_):
    '''
    Toi même tu sais
    '''
    # TODO: handle errors
    if np.isscalar(Zenith_):
        _grammage = ComputeDistanceGrammage(Zenith_, XmaxDistance_, LongitudinalDistance_, ShowerCoreHeight_)
    else :
        _grammage = [ComputeDistanceGrammage(Zenith_[i], XmaxDistance_[i], LongitudinalDistance_[i], ShowerCoreHeight_[i]) for i in range(len(Zenith_))]
    #print(_grammage)
    return np.array(_grammage)


def ComputeSourceErrorGrammage(Grammage_, RecAzimuth_, RecZenith_, XSourceRec_, YSourceRec_, ZSourceRec_, InjectionHeight_, ShowerCoreHeight_):
    '''
    Toi même tu sais
    '''
    # TODO: handle errors
    _LongitudinalDistance = ComputeLongitudinalDistance(RecAzimuth_, RecZenith_, InjectionHeight_, ShowerCoreHeight_, XSourceRec_, YSourceRec_, ZSourceRec_)
    _grammage_error = Grammage_ - ComputeGrammage(RecZenith_, ZSourceRec_, ShowerCoreHeight_, InjectionHeight_, _LongitudinalDistance)

    return _grammage_error

def ComputeSourceErrorGrammage_alternative_method(Azimuth_, Zenith_, x_Xmax_, y_Xmax_, z_Xmax_, RecAzimuth_, RecZenith_, XSourceRec_, YSourceRec_, ZSourceRec_, InjectionHeight_, ShowerCoreHeight_, XmaxDistance_):
    '''
    Toi même tu sais
    '''
    # TODO: handle errors
    print("/! Warning !!! Simulation grammage recomputed here !")
    LongitudinalDistance_Xmax = ComputeLongitudinalDistance(Azimuth_, Zenith_, InjectionHeight_, ShowerCoreHeight_, x_Xmax_, y_Xmax_, z_Xmax_)
    _LongitudinalDistance_Source = ComputeLongitudinalDistance(RecAzimuth_, RecZenith_, InjectionHeight_, ShowerCoreHeight_, XSourceRec_, YSourceRec_, ZSourceRec_)
    _SourceDistance = np.sqrt((XSourceRec_)**2 + (YSourceRec_)**2 + (ZSourceRec_ - ShowerCoreHeight_)**2)
    _grammage_recons = ComputeGrammage(RecZenith_, _SourceDistance, ShowerCoreHeight_, InjectionHeight_, _LongitudinalDistance_Source)
    _grammage_error = ComputeGrammage(Zenith_, XmaxDistance_, ShowerCoreHeight_, InjectionHeight_, LongitudinalDistance_Xmax) - _grammage_recons
    return _grammage_recons, _grammage_error, LongitudinalDistance_Xmax, _LongitudinalDistance_Source


def ComputeInjectionPoint(Azimuth_, Zenith_, InjectionHeight_, ShowerCoreHeight_):
    '''
    Toi même tu sais
    '''
    # TODO: handle errors
    k_shower = np.array([np.cos(Azimuth_*np.pi/180.)*np.sin(Zenith_*np.pi/180),np.sin(Azimuth_*np.pi/180.)*np.sin(Zenith_*np.pi/180), np.cos(Zenith_*np.pi/180)])
    _delta = (kRearth + ShowerCoreHeight_)**2*np.cos(Zenith_*np.pi/180.)**2 + (InjectionHeight_ - ShowerCoreHeight_)*(InjectionHeight_ + ShowerCoreHeight_ + 2.*kRearth)
    _injection_length = (kRearth + ShowerCoreHeight_)*np.cos(Zenith_*np.pi/180.) + np.sqrt(_delta)
    InjectionX = - k_shower[0]*_injection_length
    InjectionY = - k_shower[1]*_injection_length
    InjectionZ = - k_shower[2]*_injection_length + ShowerCoreHeight_

    return np.array([InjectionX, InjectionY, InjectionZ])

def ComputeLongitudinalDistance(Azimuth_, Zenith_, InjectionHeight_, ShowerCoreHeight_, XSourceRec_, YSourceRec_, ZSourceRec_):
    '''
    Toi même tu sais
    '''
    # TODO: handle errors
    if np.isscalar(Zenith_):
        _L = np.linalg.norm(np.array([XSourceRec_, YSourceRec_, ZSourceRec_]) - ComputeInjectionPoint(Azimuth_, Zenith_, InjectionHeight_, ShowerCoreHeight_))
    else:
        #print(len(Zenith_))
        #_L = [np.linalg.norm(np.array([XSourceRec_[i], YSourceRec_[i], ZSourceRec_[i]]) - ComputeInjectionPoint(Azimuth_[i], Zenith_[i], InjectionHeight_[i], ShowerCoreHeight_[i])) for i in range(len(Zenith_))]
        _L = [np.linalg.norm(np.array([XSourceRec_[i], YSourceRec_[i], ZSourceRec_[i]]) - ComputeInjectionPoint(Azimuth_[i], Zenith_[i], InjectionHeight_[i], ShowerCoreHeight_[i])) for i in range(len(Zenith_))]
        

    return _L

def ComputeSourceError_Long_Lat(Azimuth_, Zenith_, x_Xmax_, y_Xmax_, z_Xmax_, XSourceRec_, YSourceRec_, ZSourceRec_):
    '''
    Toi même tu sais
    '''
    # TODO: handle errors
    k_shower = np.array([np.cos(Azimuth_*np.pi/180.)*np.sin(Zenith_*np.pi/180),np.sin(Azimuth_*np.pi/180.)*np.sin(Zenith_*np.pi/180), np.cos(Zenith_*np.pi/180)])
    if np.isscalar(Zenith_):
        _DeltaLong = np.dot(np.array([XSourceRec_ - x_Xmax_, YSourceRec_ - y_Xmax_, ZSourceRec_ - z_Xmax_]), k_shower)
        _DeltaLat = np.linalg.norm(np.cross(np.array([XSourceRec_ - x_Xmax_, YSourceRec_ - y_Xmax_, ZSourceRec_ - z_Xmax_]), k_shower))
    else:
        _DeltaLong = [np.dot(np.array([XSourceRec_[i] - x_Xmax_[i], YSourceRec_[i] - y_Xmax_[i], ZSourceRec_[i] - z_Xmax_[i]]), k_shower[:,i]) for i in range(len(Zenith_))]
        _DeltaLat = [np.linalg.norm(np.cross(np.array([XSourceRec_[i] - x_Xmax_[i], YSourceRec_[i] - y_Xmax_[i], ZSourceRec_[i] - z_Xmax_[i]]), k_shower[:,i]))  for i in range(len(Zenith_))]

    return _DeltaLong, _DeltaLat

def GetLocalZenith(Zenith_, LocalHeight_, StartHeight_):
    '''
    Compute zenith angle at any point along the erath curvature
    Inputs: Zenith_, InjectionHeight_, ShowerCoreHeight_
    Outputs: Zenith angle at given location
    '''
    # TODO: handle errors
    _delta = (kRearth + StartHeight_)**2*np.cos(Zenith_*np.pi/180.)**2 + (LocalHeight_ - StartHeight_)*(LocalHeight_ + StartHeight_ + 2.*kRearth)
    _path_length = (kRearth + StartHeight_)*np.cos(Zenith_*np.pi/180.) + np.sqrt(_delta)
    _Zenith_at = (np.pi-np.arccos((_path_length**2 + (kRearth + LocalHeight_)**2 - (kRearth + StartHeight_)**2)/(2.*_path_length*(kRearth + LocalHeight_))))*180./np.pi
    #print('zenith_at', _Zenith_at)

    return _Zenith_at

def GetLocalHeight(Zenith_, StartHeight_, PathLength_):
    _height_at =  -kRearth + np.sqrt((kRearth + StartHeight_)**2 + PathLength_**2 - 2.*PathLength_*(kRearth+StartHeight_)*np.cos(Zenith_*np.pi/180.))
    return _height_at

def GetDensity(_height,model):

    if model == "isothermal":
            #Using isothermal Model
            rho_0 = 1.225    #kg/m^3
            M = 0.028966    #kg/mol
            g = 9.81        #m.s^-2
            T = 288.        #
            R = 8.32        #J/K/mol , J=kg m2/s2
            rho = rho_0*np.exp(-g*M*_height/(R*T))  # kg/m3

    elif model == "linsley":
        bl = np.array([1222., 1144., 1305.5948, 540.1778,1])*10  # g/cm2  ==> kg/cm3
        cl = np.array([9941.8638, 8781.5355, 6361.4304, 7721.7016, 1e7])  #m
        hl = np.array([4,10,40,100,113])*1e3  #m
        #print(_height)
        #print(hl[-1])
        if _height>=hl[-1]:  # no more air
            rho = 0
        else:
            hlinf = np.array([0, 4,10,40,100])*1e3  #m
            ind = np.logical_and([_height>=hlinf],[_height<hl])[0]
            #print(ind)
            rho = bl[ind]/cl[ind]*np.exp(-_height/cl[ind])
            #print(rho, ind, _height)
            rho = rho[0]
    else:
        print("#### Error in GetDensity: model can only be isothermal or linsley.")
        return 0

    return rho

def ComputeDistanceGrammage(Zenith_, XmaxDistance_, LongitudinalDistance_, ShowerCoreHeight_):
    '''
    Toi même tu sais
    '''
    # TODO: handle errors

    X = 0.
    dl_tot = 0
    conversion_factor = 0.1 #-> kg/m^-2 -> g/cm^-2

    _height = GetLocalHeight(Zenith_, ShowerCoreHeight_, XmaxDistance_)
    _zenith = GetLocalZenith(Zenith_, _height, ShowerCoreHeight_)
    nbe_iteration = 1000
    dl = LongitudinalDistance_/nbe_iteration                  # 100 steps because no time for more
    #compute zenith at Xmax

    for dl in np.repeat(dl, nbe_iteration+1):                  #Do not start at 0...

        _height_new = GetLocalHeight(_zenith, _height, dl)
        _zenith_new = GetLocalZenith(_zenith, _height_new, _height)
        if _height_new <0: continue
        dX =  GetDensity(_height,'linsley')* dl * conversion_factor
        X += dX
        if np.isnan(X): print(LongitudinalDistance_, dl_tot, _height, _zenith, X, dX, Zenith_, ShowerCoreHeight_)
        #print(LongitudinalDistance_, dl_tot, _height_new, _zenith_new, X, dX, Zenith_, ShowerCoreHeight_)
        _height=_height_new
        _zenith = _zenith_new
        dl_tot +=dl

    return X
################################################################################
#Plots tools

def get_labels_2D(xlabel_, ylabel_, title_, ax_, legend_flag=False):
    '''
    Toi même tu sais
    '''
    # TODO: handle errors
    ax_.set_xlabel(xlabel_)
    ax_.set_ylabel(ylabel_)
    ax_.set_title(title_)
    if legend_flag : ax_.legend()

    return 0

def get_scatter_plot(figure_, axis_, x_axis_, y_axis_, xlabel_, ylabel_, title_, legend_='None', legend_flag=False, alpha_=1):
    '''
    Toi même tu sais
    '''
    # TODO: handle errors
    #figure, axis = plt.subplots()
    if legend_flag:
        axis_.scatter(x_axis_, y_axis_, label=legend_, alpha=alpha_)
    else:
        axis_.scatter(x_axis_, y_axis_, alpha=alpha_)

    # if bin_mean_flag:
    #     mean, mean_edges, mean_bins = stats.binned_statistic(x_axis_, y_axis_, statistic='mean', bins=Nbins_)
    #     std, std_edges, std_bins = stats.binned_statistic(x_axis_, y_axis_, statistic='std', bins=Nbins_)
    #     pos_step = np.diff(mean_edges)/2
    #     mean_pos = mean_edges[:-1] + pos_step
    #
    #     axis_.errorbar(mean_pos, mean, yerr=std, linestyle='--', color='black')
    #     ymin = axis_.get_ylim()[0]
    #     ymax = axis_.get_ylim()[-1]
    #     for edge in mean_edges:
    #         axis_.axvline(edge, linewidth=1, color='gray', zorder=0)
    #     #a = axis_.bar(mean_edges, ymax+10, bottom=ymin-1, linestyle=':', width=0.5, color='gray', zorder=0)
    #     axis_.set_ylim(ymin, ymax)


    get_labels_2D(xlabel_, ylabel_, title_, axis_, legend_flag)

    return 0

def get_averaged_bins(figure_, axis_, x_axis_, y_axis_, xmin_, xmax_, Nbins_, color_):

    mean, mean_edges, mean_bins = stats.binned_statistic(x_axis_, y_axis_, range=(xmin_, xmax_), statistic='mean', bins=Nbins_)
    std, std_edges, std_bins = stats.binned_statistic(x_axis_, y_axis_, range=(xmin_, xmax_), statistic='std', bins=Nbins_)
    pos_step = np.diff(mean_edges)/2
    mean_pos = mean_edges[:-1] + pos_step

    axis_.errorbar(mean_pos, mean, yerr=std, linestyle='--', marker='D', color=color_)
    ymin = axis_.get_ylim()[0]
    ymax = axis_.get_ylim()[-1]
    for edge in mean_edges:
        axis_.axvline(edge, linewidth=1, color='gray', zorder=0)
    axis_.set_ylim(ymin, ymax)

    return 0

def get_color_scatter_plot(figure_, axis_, x_axis_, y_axis_, color_axis_, xlabel_, ylabel_, title_, bin_mean_flag=False, Nbins_=0, legend_='None', legend_flag=False):
    '''
    Toi même tu sais
    '''
    # TODO: handle errors
    #figure, axis = plt.subplots()
    if legend_flag:
        axis_.scatter(x_axis_, y_axis_, c=color_axis_, label=legend_)
    else:
        axis_.scatter(x_axis_, y_axis_, c=color_axis_)

    if bin_mean_flag:
        mean, mean_edges, mean_bins = stats.binned_statistic(x_axis_, y_axis_, statistic='mean', bins=Nbins_)
        std, std_edges, std_bins = stats.binned_statistic(x_axis_, y_axis_, statistic='std', bins=Nbins_)
        axis_.errorbar(mean_edges, mean, yerr=std, c=color_axis_, label='mean')

    get_labels_2D(xlabel_, ylabel_, title_, axis_, legend_flag)

    return 0


def get_histo_plot(figure_, axis_, x_axis_, xlabel_, ylabel_, title_, legend_='None', legend_flag=False, Alpha=1):
    '''
    Toi même tu sais
    '''
    # TODO: handle errors
    #figure, axis = plt.subplots()
    if legend_flag:
        #axis_.hist(x_axis_, bins=4*int(np.sqrt(len(x_axis_))), range=(-0.0001,0.01), alpha=Alpha, label=legend_)
        axis_.hist(x_axis_, bins=4*int(np.sqrt(len(x_axis_))), alpha=Alpha, label=legend_)
    else:
        axis_.hist(x_axis_, bins=4*int(np.sqrt(len(x_axis_))), alpha=Alpha)

    get_labels_2D(xlabel_, ylabel_, title_, axis_, legend_flag)

    return 0
