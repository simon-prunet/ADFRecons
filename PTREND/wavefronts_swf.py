import numpy as np
from numba import njit, float64, prange
import utils as utils
from iminuit import cost, Minuit
import physical_parameters as phys_params

kwd = {"fastmath": {"reassoc", "contract", "arcp"}}


@njit(**kwd)
def RefractionIndexAtPosition(X):

    R2 = X[0]*X[0] + X[1]*X[1]
    h = (np.sqrt((X[2]+phys_params.R_earth)**2 + R2) - phys_params.R_earth)/1e3 # Altitude in km
    rh = phys_params.ns*np.exp(phys_params.kr*h)
    n = 1.+1e-6*rh
    return (n)


#@njit(**kwd)
def ZHSEffectiveRefractionIndex(X0, Xa):

    R02 = X0[0]**2 + X0[1]**2

    # Altitude of emission in km
    h0 = (np.sqrt((X0[2]+phys_params.R_earth)**2 + R02) - phys_params.R_earth)/1e3
    # print('Altitude of emission in km = ', h0)
    # print(h0)
    # Refractivity at emission 
    rh0 = phys_params.ns*np.exp(phys_params.kr*h0)

    modr = np.sqrt(R02)
    # print(modr)

    if (modr > 1e3):

        # Vector between antenna and emission point
        U = Xa-X0
        # Divide into pieces shorter than 10km
        nint = np.int32(modr/2e4)+1
        K = U/nint

        # Current point coordinates and altitude
        Curr = X0
        currh = h0
        s = 0.

        for i in np.arange(nint):
            Next = Curr + K  # Next point
            nextR2 = Next[0]*Next[0] + Next[1]*Next[1]
            nexth = (np.sqrt((Next[2]+phys_params.R_earth)**2 + nextR2 ) - phys_params.R_earth)/1e3
            if (np.abs(nexth-currh) > 1e-10):
                s += (np.exp(phys_params.kr*nexth)-np.exp(phys_params.kr*currh))/(phys_params.kr*(nexth-currh))
            else:
                s += np.exp(phys_params.kr*currh)

            Curr = Next
            currh = nexth
            # print (currh)

        avn = phys_params.ns*s/nint
        # print(avn)
        n_eff = 1. + 1e-6*avn # Effective (average) index

    else:
        # without numerical integration
        hd = Xa[2]/1e3 # Antenna altitude
        #if (np.abs(hd-h0) > 1e-10):
        avn = (phys_params.ns/(phys_params.kr*(hd-h0)))*(np.exp(phys_params.kr*hd)-np.exp(phys_params.kr*h0))
        #else:
        #    avn = ns*np.exp(kr*h0)
        n_eff = 1. + 1e-6*avn # Effective (average) index

    return (n_eff)


# SWF related functions
def SWF_model_xyz(Xants, x_eff, y_eff, z_eff, t_s):

    x, y, z = Xants
    nants = Xants.shape[1]
    tants = np.zeros(nants)
    for i in range(nants):
        n_average = ZHSEffectiveRefractionIndex([x_eff, y_eff, z_eff], Xants.T[i, :])
        tants[i] = t_s + n_average * np.sqrt((x[i]-x_eff)**2 + (y[i]-y_eff)**2 + (z[i]-z_eff)**2) / phys_params.c_light

    return tants


def SWF_model_xyz_v2(Xants, x_eff, y_eff, z_eff):

    x, y, z = Xants
    nants = Xants.shape[1]
    delta_tants = np.zeros(nants)
    n_average_0 = ZHSEffectiveRefractionIndex([x_eff, y_eff, z_eff], Xants.T[0, :])
    for i in range(nants):
        n_average = ZHSEffectiveRefractionIndex([x_eff, y_eff, z_eff], Xants.T[i, :])
        delta_tants[i] = 1.0/phys_params.c_light * ( n_average * np.sqrt((x[i]-x_eff)**2 + (y[i]-y_eff)**2 + (z[i]-z_eff)**2) - n_average_0*np.sqrt((x[0]-x_eff)**2 + (y[0]-y_eff)**2 + (z[0]-z_eff)**2)) 

    return delta_tants


def SWF_simulation_xyz(Xants, x_eff, y_eff, z_eff, t_s, sigma_t=5e-9):
    tants = SWF_model_xyz(Xants, x_eff, y_eff, z_eff, t_s)
    n = np.random.standard_normal(tants.size) * sigma_t
    return (tants + n)


def get_SWF_fit(x_ants, t_ants, initial_guess, sigma_t=5e-9, ncall=100):
    x = x_ants[:, 0]
    y = x_ants[:, 1]
    z = x_ants[:, 2]

    leastsquares = cost.LeastSquares((x, y, z), t_ants, t_ants*0 + sigma_t, SWF_model_xyz)
    leastsquares._ndim = 3
    x0 = initial_guess[0]
    y0 = initial_guess[1]
    z0 = initial_guess[2]
    t0 = initial_guess[3]
    m = Minuit(leastsquares, x_eff=x0, y_eff=y0, z_eff=z0, t_s=t0)
    m.migrad(ncall=ncall)
    return np.array(m.values)


def get_SWF_fit_v2(x_ants, t_ants, initial_guess, sigma_t=5e-9, ncall=100):
    x = x_ants[:, 0]
    y = x_ants[:, 1]
    z = x_ants[:, 2]

    delta_tants = t_ants - t_ants[0]

    leastsquares = cost.LeastSquares((x, y, z), delta_tants, t_ants*0 + sigma_t * np.sqrt(2), SWF_model_xyz_v2)
    leastsquares._ndim = 3
    x0 = initial_guess[0]
    y0 = initial_guess[1]
    z0 = initial_guess[2]

    m = Minuit(leastsquares, x_eff=x0, y_eff=y0, z_eff=z0)
    m.migrad(ncall=ncall)
    return np.array(m.values)
