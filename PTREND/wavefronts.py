import numpy as np
from numba import njit, float64
from scipy.spatial.transform import Rotation as R
from scipy.optimize import fsolve

R_earth = 6371007.0
ns = 325
kr = -0.1218
groundAltitude = 1086.0
B_dec = 0.
B_inc = np.pi/2. + 1.0609856522873529
# Magnetic field direction (unit) vector
Bvec = np.array([np.sin(B_inc)*np.cos(B_dec),np.sin(B_inc)*np.sin(B_inc),np.cos(B_inc)])

# Simple numba example
@njit
def dotme(x,y,z):
    res =  np.dot(x,x)
    res += np.dot(y,y)
    res += np.dot(z,z)
    return(res)

@njit
def RefractionIndexAtPosition(X):

    R2 = X[0]*X[0] + X[1]*X[1]
    h = (np.sqrt( (X[2]+R_earth)**2 + R2 ) - R_earth)/1e3 # Altitude in km
    rh = ns*np.exp(kr*h)
    n = 1.+1e-6*rh-1
    return (n)

@njit
def ZHSEffectiveRefractionIndex(X0,Xa):

    R02 = X0[0]**2 + X0[1]**2
    
    # Altitude of emission in km
    h0 = (np.sqrt( (X0[2]+R_earth)**2 + R02 ) - R_earth)/1e3
    # print(h0)
    
    # Refractivity at emission 
    rh0 = ns*np.exp(kr*h0)

    modr = np.sqrt(R02)
    # print(modr)

    if (modr > 1e3):

        # Vector between antenna and emission point
        U = Xa-X0
        # Divide into pieces shorter than 10km
        nint = np.int(modr/2e4)+1
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


def compute_Cerenkov(eta, K, xmaxDist, Xmax, delta, groundAltitude):

    '''
    Solve for Cerenkov angle by minimizing
    time delay between light rays originating from Xb and Xmax and arriving
    at observer's position. 
    eta:   azimuth of observer's position in shower plane coordinates 
    K:     direction vector of shower
    Xmax:  coordinates of Xmax point
    delta: distance between Xmax and Xb points
    groundAltitude: self explanatory

    Returns:     
    omega: angle between shower direction and line joining Xmax and observer's position

    '''

    # Compute coordinates of point before Xmax
    Xb = Xmax - delta*K
    # Projected shower direction in horizontal plane
    nk2D = np.sqrt(K[0]*K[0]+K[1]*K[1])
    K_plan = np.array([K[0]/nk2D,K[1]/nk2D,0.])
    # Direction vector to observer's position in horizontal plane
    # This assumes all observers positions are in the horizontal plane
    ce = np.cos(eta); se=np.sin(eta)
    U = np.array([ce*K_plan[0]+se*K_plan[1],-se*K_plan[0]+ce*K_plan[1],0.])
    # Compute angle between shower direction and (horizontal) direction to observer
    alpha = np.acos(np.dot(K,U))


    def compute_delay(omega):

        X = compute_observer_position(omega)
        n0 = ZHSEffectiveRefractionIndex(Xmax,X)
        n1 = ZHSEffectiveRefractionIndex(Xb  ,X)
        res = minor_equation(omega,n0,n1)

    @njit
    def compute_observer_position(omega):
        '''
        Given angle between shower direction (K) and line joining Xmax and observer's position,
        horizontal direction to observer's position, Xmax position and groundAltitude, compute
        coordinates of observer
        '''

        # Compute rotation axis. Make sure it is normalized
        Rot_axis = np.cross(U,K)
        Rot_axis /= np.linalg.norm(Rot_axis)
        # Define rotation using scipy's method
        Rotation = R.from_rotvec(omega * Rot_axis)
        Dir_obs  = Rotation.apply(K)
        # Compute observer's position
        t = (groundAltitude - Xmax[2])/Dir_obs[2]
        X = Xmax + t*Dir_obs
        return (X)

    @njit
    def minor_equation(omega, n0, n1):

        '''
        Compute time delay (in m)
        '''
        Lx = xmaxDist
        sa = np.sin(alpha)
        saw = np.sin(alpha+omega) # Note the difference with Valentin Decoene's code... To be confirmed.
        com = np.cos(omega)
        # Eq. 3.38 p125.
        res = Lx*Lx * sa*sa *(n0*n0-n1*n1) + 2*Lx*sa*saw*delta*(n0-n1*n1*com) + delta*delta*(1.-n1*n1)*saw*saw

        return(res)


    # Now solve for omega
    # Starting point at standard value acos(1/n(Xmax)) 
    omega_cr_guess = np.acos(1./RefractionIndexAtPosition(Xmax))
    omega_cr = fsolve(compute_delay,[omega_cr_guess])

    return(omega_cr)


# Loss functions (chi2), according to different models:
# PWF: Plane wave function
# SWF: Spherical wave function
# ADF: 

@njit
def PWF_loss(Xants, tants, theta, phi, cr=1.0):
    '''
    Defines Chi2 by summing model residuals
    over antenna pairs (i,j):
    loss = 0.5 \sum_{i>j} ((Xants[i,:]-Xants[j,:]).K - cr(tants[i]-tants[j]))**2
    where:
    Xants are the antenna positions (shape=(nants,3))
    tants are the antenna arrival times of the wavefront (trigger time, shape=(nants,))
    theta, phi: spherical coordinates of unit shower direction vector K
    cr is radiation speed, by default 1 since time is expressed in m.
    '''

    nants = shape(tants)[0]
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)
    K = np.array([st*cp,st*sp,ct])
    # Make sure tants and Xants are compatible
    if (Xants.shape[0] != nants):
        print("Shapes of tants and Xants are incompatible",tants.shape,Xants.shape)
        return None
    tmp = 0.
    for j in range(nants):
        for i in range(j+1,nants):
            res = np.dot(Xants[j,:]-Xants[i,:],K)-cr*(tants[j]-tants[i])
            tmp += res*res
    chi2 = 0.5*tmp
    return (chi2)


def SWF_loss(Xants, tants, theta, phi, r_xmax, t_s, cr=1.0):

    '''
    Defines Chi2 by summing model residuals over antennas  (i):
    loss = 0.5 \sum_i ( cr(tants[i]-t_s) - \sqrt{(Xants[i,0]-x_s)**2)+(Xants[i,1]-y_s)**2+(Xants[i,2]-z_s)**2} )**2
    where:
    Xants are the antenna positions (shape=(nants,3))
    tants are the trigger times (shape=(nants,))
    x_s = \sin(\theta)\cos(\phi)
    y_s = \sin(\theta)\sin(\phi)
    z_s = \cos(\theta)
    \theta, \phi are the spherical coordinates of the vector K
    t_s is the source emission time
    cr is the radiation speed in medium, by default 1 since time is expressed in m.
    '''

    nants = shape(tants)[0]
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)
    K = np.array([st*cp,st*sp,ct])
    Xmax = -r_xmax * K + np.array([0.,0.,groundAltitude]) # Xmax is in the opposite direction to shower propagation.

    # Make sure Xants and tants are compatible
    if (Xants.shape[0] != nants):
        print("Shapes of tants and Xants are incompatible",tants.shape, Xants.shape)
        return None
    tmp = 0.
    for i in range(nants):
        # Compute average refraction index between emission and observer
        n_average = ZHSEffectiveRefractionIndex(Xmax, Xants[i,:])
        dX = Xants[i,:] - Xmax
        # Spherical wave front
        res = cr*(tants[i]-t_s) - n_average*np.linalg.norm(dX)
        tmp += res*res

    chi2 = 0.5*tmp
    return(chi2)

def ADF_loss(Aants, Xants, Xmax, theta, phi, delta_omega, r_xmax, asym_coeff=0.01):
    
    '''

    Defines Chi2 by summing *amplitude* model residuals over antennas (i):
    loss = \sum_i (A_i - f_i^{ADF}(\theta,\phi,\delta\omega,A,r_xmax))**2
    where the ADF function reads:
    
    f_i = f_i(\omega_i, \eta_i, \alpha, l_i, \delta_omega, A)
        = A/l_i f_geom(\alpha, \eta_i) f_Cerenkov(\omega,\delta_\omega)
    
    where 
    
    f_geom(\alpha, \eta_i) = (1 + B \sin(\alpha))**2 \cos(\eta_i) # B is here the geomagnetic asymmetry
    f_Cerenkov(\omega_i,\delta_\omega) = 1 / (1+4{ (\tan(\omega_i)/\tan(\omega_c))**2 - 1 ) / \delta_\omega }**2 )
    
    Input parameters are: \theta, \phi define the shower direction angles, \delta_\omega the width of the Cerenkov ring, 
    A is the amplitude paramter, r_xmax is the norm of the position vector at Xmax.

    Derived parameters are: 
    \alpha, angle between the shower axis and the magnetic field
    \eta_i is the azimuthal angle of the (projection of the) antenna position in shower plane
    \omega_i is the angle between the shower axis and the vector going from Xmax to the antenna position

    '''

    nants = Aants.shape[0]
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)
    # Define shower basis vectors
    K = np.array([st*cp,st*sp,ct])
    KxB = np.cross(K,Bvec); KxB /= np.linalg.norm(KxB)
    KxKxB = np.cross(K,KxB); KxKxB /= np.linalg.norm(KxKxB)
    # 
    XmaxDist = (groundAltitude-Xmax[2])/K[2]
    asym = asym_coeff * (1. - np.dot(K,Bvec)**2) # Azimuthal dependence, in \sin^2(\eta)
    ### TBC


    

