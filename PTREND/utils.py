import numpy as np
from numba import njit

kwd = {"fastmath": {"reassoc", "contract", "arcp"}}
def thetaphi_to_k(theta, phi):
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    K =  - np.array([st*cp, st*sp, ct])
    return K

def k_to_theta_phi(k):
    theta = np.arccos(-k[2])
    cp = - k[0]/np.sin(theta)
    sp = - k[1]/np.sin(theta)
    phi = np.arctan2(sp, cp)
    if phi < 0:
        phi += 2 * np.pi
    return theta, phi


def paramsswf_to_xyz(theta, phi, r_xmax, array_center):
    K = thetaphi_to_k(theta, phi)
    xyz = -r_xmax * K + array_center
    return xyz


def xyz_to_paramsswf(xyz, array_center):
    Kp = xyz - array_center
    r_xmax = np.sqrt(np.dot(Kp, Kp))
    K = - Kp/r_xmax
    theta, phi = k_to_theta_phi(K)
    return theta, phi, r_xmax


@njit(**kwd)
def RefractionIndexAtPosition(X):

    R2 = X[0]*X[0] + X[1]*X[1]
    h = (np.sqrt( (X[2]+R_earth)**2 + R2 ) - R_earth)/1e3 # Altitude in km
    rh = ns*np.exp(kr*h)
    n = 1.+1e-6*rh
    return (n)

@njit(**kwd)
def ZHSEffectiveRefractionIndex(X0, Xa):

    R02 = X0[0]**2 + X0[1]**2
    
    # Altitude of emission in km
    h0 = (np.sqrt( (X0[2]+R_earth)**2 + R02 ) - R_earth)/1e3
    # print('Altitude of emission in km = ', h0)
    # print(h0)
    
    # Refractivity at emission 
    rh0 = ns*np.exp(kr*h0)

    modr = np.sqrt(R02)
    # print(modr)

    if (modr > 1e3):

        # Vector between antenna and emission point
        U = Xa-X0
        # Divide into pieces shorter than 10km
        nint = np.int32(modr/2e4)+1
        K = U/nint

        # Current point coordinates and altitude
        Curr  = X0
        currh = h0
        s = 0.

        for i in np.arange(nint):
            Next = Curr + K # Next point
            nextR2 = Next[0]*Next[0] + Next[1]*Next[1]
            nexth  = (np.sqrt( (Next[2]+R_earth)**2 + nextR2 ) - R_earth)/1e3
            if (np.abs(nexth-currh) > 1e-10):
                s += (np.exp(kr*nexth)-np.exp(kr*currh))/(kr*(nexth-currh))
            else:
                s += np.exp(kr*currh)

            Curr = Next
            currh = nexth
            # print (currh)

        avn = ns*s/nint
        # print(avn)
        n_eff = 1. + 1e-6*avn # Effective (average) index

    else:

        # without numerical integration
        hd = Xa[2]/1e3 # Antenna altitude
        #if (np.abs(hd-h0) > 1e-10):
        avn = (ns/(kr*(hd-h0)))*(np.exp(kr*hd)-np.exp(kr*h0))
        #else:
        #    avn = ns*np.exp(kr*h0)

        n_eff = 1. + 1e-6*avn # Effective (average) index

    return (n_eff)