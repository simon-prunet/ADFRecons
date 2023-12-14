import numpy as np
from numba import njit, float64, prange
from scipy.spatial.transform import Rotation as R
from scipy.optimize import fsolve
from solver import newton
from rotation import rotation
# Used for interpolation
n_omega_cr = 20

# Physical constants
c_light = 2.997924580e8
R_earth = 6371007.0
ns = 325
kr = -0.1218
groundAltitude = 1086.0
B_dec = 0.
B_inc = np.pi/2. + 1.0609856522873529
# Magnetic field direction (unit) vector
Bvec = np.array([np.sin(B_inc)*np.cos(B_dec),np.sin(B_inc)*np.sin(B_dec),np.cos(B_inc)])

kwd = {"fastmath": {"reassoc", "contract", "arcp"}}

# Simple numba example
@njit(**kwd)
def dotme(x,y,z):
    res =  np.dot(x,x)
    res += np.dot(y,y)
    res += np.dot(z,z)
    return(res)

@njit(**kwd)
def RefractionIndexAtPosition(X):

    R2 = X[0]*X[0] + X[1]*X[1]
    h = (np.sqrt( (X[2]+R_earth)**2 + R2 ) - R_earth)/1e3 # Altitude in km
    rh = ns*np.exp(kr*h)
    n = 1.+1e-6*rh
    return (n)

@njit(**kwd)
def ZHSEffectiveRefractionIndex(X0,Xa):

    R02 = X0[0]**2 + X0[1]**2
    
    # Altitude of emission in km
    h0 = (np.sqrt( (X0[2]+R_earth)**2 + R02 ) - R_earth)/1e3
    # print('Altitude of emission in km = ',h0)
    # print(h0)
    
    # Refractivity at emission 
    rh0 = ns*np.exp(kr*h0)

    modr = np.sqrt(R02)
    # print(modr)

    if (modr > 1e3):

        # Vector between antenna and emission point
        U = Xa-X0
        # Divide into pieces shorter than 10km
        #nint = np.int(modr/2e4)+1
        nint = int(modr/2e4)+1
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

@njit(**kwd)
def compute_observer_position(omega,Xmax,U,K):
    '''
    Given angle between shower direction (K) and line joining Xmax and observer's position,
    horizontal direction to observer's position, Xmax position and groundAltitude, compute
    coordinates of observer
    '''

    # Compute rotation axis. Make sure it is normalized
    Rot_axis = np.cross(U,K)
    Rot_axis /= np.linalg.norm(Rot_axis)
    # Compute rotation matrix from Rodrigues formula
    Rotmat = rotation(-omega,Rot_axis)
    # Define rotation using scipy's method
    # Rotation = R.from_rotvec(-omega * Rot_axis)
    # print('#####')
    # print(Rotation.as_matrix())
    # print('#####')
    # Dir_obs  = Rotation.apply(K)
    Dir_obs = np.dot(Rotmat,K)
    # Compute observer's position
    t = (groundAltitude - Xmax[2])/Dir_obs[2]
    X = Xmax + t*Dir_obs
    return (X)

@njit(**kwd)
def minor_equation(omega, n0, n1, alpha, delta, xmaxDist):

    '''
    Compute time delay (in m)
    '''
    Lx = xmaxDist
    sa = np.sin(alpha)
    saw = np.sin(alpha-omega) # Keeping minus sign to compare to Valentin's results. Should be plus sign.
    com = np.cos(omega)
    # Eq. 3.38 p125.
    res = Lx*Lx * sa*sa *(n0*n0-n1*n1) + 2*Lx*sa*saw*delta*(n0-n1*n1*com) + delta*delta*(1.-n1*n1)*saw*saw

    return(res)

@njit(**kwd)
def compute_delay(omega,Xmax,Xb,U,K,alpha,delta, xmaxDist):

    X = compute_observer_position(omega,Xmax,U,K)
    # print('omega = ',omega,'X_obs = ',X)
    n0 = ZHSEffectiveRefractionIndex(Xmax,X)
    # print('n0 = ',n0)
    n1 = ZHSEffectiveRefractionIndex(Xb  ,X)
    # print('n1 = ',n1)
    res = minor_equation(omega,n0,n1,alpha, delta, xmaxDist)
    # print('delay = ',res)
    return(res)



@njit(**kwd)
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
    alpha = np.arccos(np.dot(K,U))


    # Now solve for omega
    # Starting point at standard value acos(1/n(Xmax)) 
    omega_cr_guess = np.arccos(1./RefractionIndexAtPosition(Xmax))
    # print("###############")
    # omega_cr = fsolve(compute_delay,[omega_cr_guess])
    omega_cr = newton(compute_delay, omega_cr_guess, args=(Xmax,Xb,U,K,alpha,delta, xmaxDist),verbose=False)
    ### DEBUG ###
    # omega_cr = omega_cr_guess
    return(omega_cr)


# @njit(**kwd)
# def compute_Cerenkov_constraint(eta, K, xmaxDist, Xmax, delta, groundAltitude):



# Loss functions (chi2), according to different models:
# PWF: Plane wave function
# SWF: Spherical wave function
# ADF: Amplitude Distribution Function (see Valentin Decoene's thesis)


def PWF_loss(params, Xants, tants, verbose=False, cr=1.0):
    '''
    Defines Chi2 by summing model residuals
    over antenna pairs (i,j):
    loss = \sum_{i>j} ((Xants[i,:]-Xants[j,:]).K - cr(tants[i]-tants[j]))**2
    where:
    params=(theta, phi): spherical coordinates of unit shower direction vector K
    Xants are the antenna positions (shape=(nants,3))
    tants are the antenna arrival times of the wavefront (trigger time, shape=(nants,))
    cr is radiation speed, by default 1 since time is expressed in m.
    '''

    theta,phi = params
    nants = tants.shape[0]
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)
    K = np.array([st*cp,st*sp,ct])
    # Make sure tants and Xants are compatible
    if (Xants.shape[0] != nants):
        print("Shapes of tants and Xants are incompatible",tants.shape,Xants.shape)
        return None
    # Use numpy outer methods to build matrix X_ij = x_i -x_j
    xk = np.dot(Xants,K)
    DXK = np.subtract.outer(xk,xk)
    DT  = np.subtract.outer(tants,tants)
    chi2 = ( (DXK - cr*DT)**2 ).sum() / 2. # Sum over upper triangle, diagonal is zero because of antisymmetry of DXK, DT
    if verbose:
        print("params = ",np.rad2deg(params))
        print("Chi2 = ",chi2)
    return(chi2)

def PWF_alternate_loss(params, Xants, tants, verbose=False, cr=1.0):
    '''
    Defines Chi2 by summing model residuals over individual antennas,
    after maximizing likelihood over reference time.
    '''
    theta,phi = params
    nants = tants.shape[0]
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)
    K = np.array([st*cp,st*sp,ct])
    # Make sure tants and Xants are compatible
    if (Xants.shape[0] != nants):
        print("Shapes of tants and Xants are incompatible",tants.shape,Xants.shape)
        return None
    xk = np.dot(Xants, K)
    ct0 = 1./nants * (cr*tants-xk).sum()
    chi2 = ((cr*tants-ct0-xk)**2).sum()
    return(chi2)

def PWF_grad(params, Xants, tants, verbose=False, cr=1.0):

    '''
    Gradient of PWF_loss, with respect to theta, phi
    '''
    theta, phi = params
    nants = tants.shape[0]
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)
    K = np.array([st*cp,st*sp,ct])

    xk = np.dot(Xants,K)
    # Use numpy outer method to build matrix X_ij = x_i - x_j
    DXK = np.subtract.outer(xk,xk)
    DT  = np.subtract.outer(tants,tants)
    RHS = DXK-cr*DT

    # Derivatives of K w.r.t. theta, phi
    dKdtheta = np.array([ct*cp,ct*sp,-st])
    dKdphi   = np.array([-st*sp,st*cp,0.])
    xk_theta = np.dot(Xants,dKdtheta)
    xk_phi   = np.dot(Xants,dKdphi)
    # Use numpy outer method to build matrix X_ij = x_i - x_j
    DXK_THETA = np.subtract.outer(xk_theta,xk_theta)
    DXK_PHI   = np.subtract.outer(xk_phi,xk_phi)

    jac_theta = np.sum(DXK_THETA*RHS) # Factor of 2 of derivatives compensates ratio of sum to upper diag sum
    jac_phi   = np.sum(DXK_PHI*RHS)
    if verbose:
        print("Jacobian = ",jac_theta,jac_phi)
    return np.array([jac_theta,jac_phi])

def PWF_hess(params, Xants, tants, verbose=False, cr=1.0):
    '''
    Hessian of PWF_loss, with respect to theta, phi
    '''
    theta, phi = params
    nants = tants.shape[0]
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)
    K = np.array([st*cp,st*sp,ct])

    xk = np.dot(Xants,K)
    # Use numpy outer method to build matrix X_ij = x_i - x_j
    DXK = np.subtract.outer(xk,xk)
    DT  = np.subtract.outer(tants,tants)
    RHS = DXK-cr*DT

    # Derivatives of K w.r.t. theta, phi
    dK_dtheta = np.array([ct*cp,ct*sp,-st])
    dK_dphi   = np.array([-st*sp,st*cp,0.])
    d2K_dtheta= np.array([-st*cp,-st*sp,-ct])
    d2K_dphi  = np.array([-st*cp,-st*sp,0.])
    d2K_dtheta_dphi = np.array([-ct*sp,ct*cp,0.]) 

    xk_theta = np.dot(Xants,dK_dtheta)
    xk_phi   = np.dot(Xants,dK_dphi)
    xk2_theta = np.dot(Xants,d2K_dtheta)
    xk2_phi   = np.dot(Xants,d2K_dphi)
    xk2_theta_phi = np.dot(Xants,d2K_dtheta_dphi)

    #Use numpy outer method to buid matrix X_ij = x_i - x_j
    DXK_THETA = np.subtract.outer(xk_theta,xk_theta)
    DXK_PHI   = np.subtract.outer(xk_phi,xk_phi)
    DXK2_THETA = np.subtract.outer(xk2_theta,xk2_theta)
    DXK2_PHI   = np.subtract.outer(xk2_phi,xk2_phi)
    DXK2_THETA_PHI = np.subtract.outer(xk2_theta_phi,xk2_theta_phi)

    hess_theta2 = np.sum(DXK2_THETA*RHS + DXK_THETA**2)
    hess_phi2   = np.sum(DXK2_PHI*RHS + DXK_PHI**2)
    hess_theta_phi = np.sum(DXK2_THETA_PHI*RHS + DXK_THETA*DXK_PHI)

    return (np.array([[hess_theta2, hess_theta_phi], [hess_theta_phi, hess_phi2]]))


###################################################
# This one is slower and not used anymore
@njit(**kwd)
def PWF_loss_nonp(params, Xants, tants, verbose=False, cr=1.0):
    '''
    Defines Chi2 by summing model residuals
    over antenna pairs (i,j):
    loss = \sum_{i>j} ((Xants[i,:]-Xants[j,:]).K - cr(tants[i]-tants[j]))**2
    where:
    params=(theta, phi): spherical coordinates of unit shower direction vector K
    Xants are the antenna positions (shape=(nants,3))
    tants are the antenna arrival times of the wavefront (trigger time, shape=(nants,))
    cr is radiation speed, by default 1 since time is expressed in m.
    '''

    theta,phi = params
    nants = tants.shape[0]
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)
    K = np.array([st*cp,st*sp,ct])
    # Make sure tants and Xants are compatible
    if (Xants.shape[0] != nants):
        print("Shapes of tants and Xants are incompatible",tants.shape,Xants.shape)
        return None
    tmp = 0.
    for j in range(nants-1):
        for i in range(j+1,nants):
            res = np.dot(Xants[j,:]-Xants[i,:],K)-cr*(tants[j]-tants[i])
            tmp += res*res
    chi2 = tmp
    if verbose:
        print("Chi2 = ",chi2)
    return (chi2)
###################################################

@njit(**kwd,parallel=False)
def SWF_loss(params, Xants, tants, verbose=False, log = False, cr=1.0):

    '''
    Defines Chi2 by summing model residuals over antennas  (i):
    loss = \sum_i ( cr(tants[i]-t_s) - \sqrt{(Xants[i,0]-x_s)**2)+(Xants[i,1]-y_s)**2+(Xants[i,2]-z_s)**2} )**2
    where:
    Xants are the antenna positions (shape=(nants,3))
    tants are the trigger times (shape=(nants,))
    x_s = \sin(\theta)\cos(\phi)
    y_s = \sin(\theta)\sin(\phi)
    z_s = \cos(\theta)

    Inputs: params = theta, phi, r_xmax, t_s
    \theta, \phi are the spherical coordinates of the vector K
    t_s is the source emission time
    cr is the radiation speed in medium, by default 1 since time is expressed in m.
    '''


    if (log is True):
        # Pass r_xmax+t_s, np.log10(r_xmax - t_s) instead of r_xmax, t_s
        theta, phi, sm, logdf = params
        df = 10.**logdf
        r_xmax = (df+sm)/2.
        t_s    = (-df+sm)/2.
    else:
        theta, phi, r_xmax, t_s = params
    nants = tants.shape[0]
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
        #if (verbose) :
        #    print('n_average = ',n_average)
        ## n_average = 1.0 #DEBUG
        dX = Xants[i,:] - Xmax
        # Spherical wave front
        res = cr*(tants[i]-t_s) - n_average*np.linalg.norm(dX)
        tmp += res*res

    chi2 = tmp
    if (verbose):
        print("theta,phi,r_xmax,t_s = ",theta,phi,r_xmax,t_s)
        print ("Chi2 = ",chi2)
    return(chi2)

@njit(**kwd,parallel=False)
def log_SWF_loss(params, Xants, tants, verbose=False, cr=1.0):
    return np.log10(SWF_loss(params,Xants,tants,verbose=verbose,cr=1.0))


@njit(**kwd)
def SWF_grad(params, Xants, tants, verbose=False, cr=1.0):
    '''
    Gradient of SWF_loss, w.r.t. theta, phi, r_xmax and t_s
    Note that this gradient is approximate in the sense that it 
    does not take into account the variations of the line of sight
    mean refractive index with Xmax(theta,phi,r_xmax)
    '''
    theta, phi, r_xmax, t_s = params
    # print("theta,phi,r_xmax,t_s = ",theta,phi,r_xmax,t_s)
    nants = tants.shape[0]
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)
    K = np.array([st*cp,st*sp,ct])
    Xmax = -r_xmax * K + np.array([0.,0.,groundAltitude]) # Xmax is in the opposite direction to shower propagation.
    # Derivatives of Xmax, w.r.t. theta, phi, r_xmax
    dK_dtheta = np.array([ct*cp,ct*sp,-st])
    dK_dphi   = np.array([-st*sp,st*cp,0.])
    dXmax_dtheta = -r_xmax*dK_dtheta
    dXmax_dphi   = -r_xmax*dK_dphi
    dXmax_drxmax = -K
    
    jac = np.zeros(4)
    for i in range(nants):
        n_average = ZHSEffectiveRefractionIndex(Xmax, Xants[i,:])
        ## n_average = 1.0 ## DEBUG
        dX = Xants[i,:] - Xmax
        ndX = np.linalg.norm(dX)
        res = cr*(tants[i]-t_s) - n_average*ndX
        # Derivatives w.r.t. theta, phi, r_xmax, t_s
        jac[0] += -2*n_average*np.dot(-dXmax_dtheta,dX)/ndX * res
        jac[1] += -2*n_average*np.dot(-dXmax_dphi,  dX)/ndX * res
        jac[2] += -2*n_average*np.dot(-dXmax_drxmax,dX)/ndX * res
        jac[3] += -2*cr                                     * res 
    if (verbose):
        print ("Jacobian = ",jac)
    return(jac)


@njit(**kwd)
def SWF_hess(params, Xants, tants, verbose=False, cr=1.0):
    '''
    Hessian of SWF loss, w.r.t. theta, phi, r_xmax, t_s
    '''
    theta, phi, r_xmax, t_s = params
    # print("theta,phi,r_xmax,t_s = ",theta,phi,r_xmax,t_s)
    nants = tants.shape[0]
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)
    K = np.array([st*cp,st*sp,ct])
    Xmax = -r_xmax * K + np.array([0.,0.,groundAltitude]) # Xmax is in the opposite direction to shower propagation.
    # Derivatives of Xmax, w.r.t. theta, phi, r_xmax
    dK_dtheta = np.array([ct*cp,ct*sp,-st])
    dK_dphi   = np.array([-st*sp,st*cp,0.])
    dXmax_dtheta = -r_xmax*dK_dtheta
    dXmax_dphi   = -r_xmax*dK_dphi
    dXmax_drxmax = -K
    ### TO BE WRITTEN... WORTH IT ?

@njit(**kwd)
def ADF_loss(params, Aants, Xants, Xmax, asym_coeff=0.01,verbose=False):
    
    '''

    Defines Chi2 by summing *amplitude* model residuals over antennas (i):
    loss = \sum_i (A_i - f_i^{ADF}(\theta,\phi,\delta\omega,A,r_xmax))**2
    where the ADF function reads:
    
    f_i = f_i(\omega_i, \eta_i, \alpha, l_i, \delta_omega, A)
        = A/l_i f_geom(\alpha, \eta_i) f_Cerenkov(\omega,\delta_\omega)
    
    where 
    
    f_geom(\alpha, \eta_i) = (1 + B \sin(\alpha))**2 \cos(\eta_i) # B is here the geomagnetic asymmetry
    f_Cerenkov(\omega_i,\delta_\omega) = 1 / (1+4{ (\tan(\omega_i)/\tan(\omega_c))**2 - 1 ) / \delta_\omega }**2 )
    
    Input parameters are: params = theta, phi, delta_omega, amplitude
    \theta, \phi define the shower direction angles, \delta_\omega the width of the Cerenkov ring, 
    A is the amplitude paramater, r_xmax is the norm of the position vector at Xmax.

    Derived parameters are: 
    \alpha, angle between the shower axis and the magnetic field
    \eta_i is the azimuthal angle of the (projection of the) antenna position in shower plane
    \omega_i is the angle between the shower axis and the vector going from Xmax to the antenna position

    '''

    theta, phi, delta_omega, amplitude = params
    nants = Aants.shape[0]
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)
    # Define shower basis vectors
    K = np.array([st*cp,st*sp,ct])
    K_plan = np.array([K[0],K[1]])
    KxB = np.cross(K,Bvec); KxB /= np.linalg.norm(KxB)
    KxKxB = np.cross(K,KxB); KxKxB /= np.linalg.norm(KxKxB)
    # Coordinate transform matrix
    mat = np.vstack((KxB,KxKxB,K))
    # 
    XmaxDist = (groundAltitude-Xmax[2])/K[2]
    
    # print('XmaxDist = ',XmaxDist)
    asym = asym_coeff * (1. - np.dot(K,Bvec)**2) # Azimuthal dependence, in \sin^2(\alpha)
    #
    # Make sure Xants and tants are compatible
    if (Xants.shape[0] != nants):
        print("Shapes of Aants and Xants are incompatible",Aants.shape, Xants.shape)
        return None

    # Precompute an array of Cerenkov angles to interpolate over (as in Valentin's code)
    omega_cerenkov = np.zeros(2*n_omega_cr+1)
    xi_table = np.arange(2*n_omega_cr+1)/n_omega_cr*np.pi
    for i in range(n_omega_cr+1):
        omega_cerenkov[i] = compute_Cerenkov(xi_table[i],K,XmaxDist,Xmax,2.0e3,groundAltitude)
    # Enforce symmetry
    omega_cerenkov[n_omega_cr+1:] = (omega_cerenkov[:n_omega_cr])[::-1]

    # Loop on antennas
    tmp = 0.
    for i in range(nants):
        # Antenna position from Xmax
        dX = Xants[i,:]-Xmax
        # Expressed in shower frame coordinates
        dX_sp = np.dot(mat,dX)
        #
        l_ant = np.linalg.norm(dX)
        eta = np.arctan2(dX_sp[1],dX_sp[0])
        omega = np.arccos(np.dot(K,dX)/l_ant)
        # vector in the plane defined by K and dX, projected onto 
        # horizontal plane
        val_plan = np.array([dX[0]/l_ant - K[0], dX[1]/l_ant - K[1]])
        # Angle between k_plan and val_plan
        xi = np.arccos(np.dot(K_plan,val_plan)
                    /np.linalg.norm(K_plan)
                    /np.linalg.norm(val_plan))
        
        # omega_cr = compute_Cerenkov(xi,K,XmaxDist,Xmax,2.0e3,groundAltitude)
        # Interpolate to save time
        omega_cr = np.interp(xi,xi_table,omega_cerenkov)
        #omega_cr = 0.015240011539221762
        # omega_cr = np.arccos(1./RefractionIndexAtPosition(Xmax))
        # print ("omega_cr = ",omega_cr)

        # Distribution width. Here rescaled by ratio of cosines (why ?)
        width = ct / (dX[2]/l_ant) * delta_omega
        #print(width)
        # Distribution
        adf = amplitude/l_ant / (1.+4.*( ((np.tan(omega)/np.tan(omega_cr))**2 - 1. )/width )**2)
        adf *= 1. + asym*np.cos(eta) # 
        # Chi2
        tmp += (Aants[i]-adf)**2

    chi2 = tmp
    if (verbose):
        print ("params = ",np.rad2deg(params[:2]),params[2:]," Chi2 = ",chi2)
    return(chi2)

@njit(**kwd)
def log_ADF_loss(params, Aants, Xants, Xmax, asym_coeff=0.01,verbose=False):

    return np.log10(ADF_loss(params, Aants, Xants, Xmax, asym_coeff=asym_coeff,verbose=verbose))


@njit(**kwd)
def PWF_residuals(params, Xants, tants, cr=1.0):

    '''
    Computes timing residuals for each antenna using plane wave model
    Note that this is defined at up to an additive constant
    '''
    theta, phi = params
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)
    K = np.array([st*cp,st*sp,ct])
    dX = Xants - np.array([0.,0.,groundAltitude])
    res = cr*tants - np.dot(dX,K)
    return(res)

@njit(**kwd)
def PWF_simulation(params, Xants, sigma_t = 5e-9, iseed=None, cr=1.0):
    '''
    Generates plane wavefront timings, zero at shower core, with jitter noise added
    '''
    theta, phi = params
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp=np.sin(phi)
    K = np.array([st*cp,st*sp,ct])
    dX = Xants - np.array([0.,0.,groundAltitude])
    tants = np.dot(dX,K) / cr 
    # Add noise
    if (iseed is not None):
        np.random.seed(iseed)
    n = np.random.standard_normal(tants.size) * sigma_t * c_light
    return (tants + n)


@njit(**kwd,parallel=False)
def SWF_residuals(params, Xants, tants, verbose=False, cr=1.0):

    '''
    Computes timing residuals for each antenna (i):
    residual[i] = ( cr(tants[i]-t_s) - \sqrt{(Xants[i,0]-x_s)**2)+(Xants[i,1]-y_s)**2+(Xants[i,2]-z_s)**2} )**2
    where:
    Xants are the antenna positions (shape=(nants,3))
    tants are the trigger times (shape=(nants,))
    x_s = \sin(\theta)\cos(\phi)
    y_s = \sin(\theta)\sin(\phi)
    z_s = \cos(\theta)

    Inputs: params = theta, phi, r_xmax, t_s
    \theta, \phi are the spherical coordinates of the vector K
    t_s is the source emission time
    cr is the radiation speed in medium, by default 1 since time is expressed in m.
    '''

    theta, phi, r_xmax, t_s = params
    # print("theta,phi,r_xmax,t_s = ",theta,phi,r_xmax,t_s)
    nants = tants.shape[0]
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)
    K = np.array([st*cp,st*sp,ct])
    Xmax = -r_xmax * K + np.array([0.,0.,groundAltitude]) # Xmax is in the opposite direction to shower propagation.

    # Make sure Xants and tants are compatible
    if (Xants.shape[0] != nants):
        print("Shapes of tants and Xants are incompatible",tants.shape, Xants.shape)
        return None
    tmp = 0.
    res = np.zeros(nants)
    for i in range(nants):
        # Compute average refraction index between emission and observer
        n_average = ZHSEffectiveRefractionIndex(Xmax, Xants[i,:])
        ## n_average = 1.0 #DEBUG
        dX = Xants[i,:] - Xmax
        # Spherical wave front
        res[i] = cr*(tants[i]-t_s) - n_average*np.linalg.norm(dX)

    return(res)

@njit(**kwd)
def SWF_simulation(params, Xants, sigma_t = 5e-9, iseed=1234, cr=1.0):
    '''
    Computes simulated wavefront timings for the spherical case.
    Inputs: params = theta, phi, r_xmax, t_s
    \theta, \phi are the spherical angular coordinates of Xmax, and  
    r_xmax is the distance of Xmax to the reference point of coordinates (0,0,groundAltitude)
    sigma_t is the timing jitter noise, in ns
    iseed is the integer random seed of the noise generator
    c_r is the speed of light in vacuum, in units of c_light
    '''
    theta, phi, r_xmax, t_s = params
    nants = Xants.shape[0]
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)
    K = np.array([st*cp, st*sp, ct])
    Xmax = -r_xmax * K + np.array([0.,0.,groundAltitude])
    tants = np.zeros(nants)
    for i in range(nants):
        n_average = ZHSEffectiveRefractionIndex(Xmax, Xants[i,:])
        dX = Xants[i,:] - Xmax
        tants[i] = t_s + n_average / cr * np.linalg.norm(dX)

    np.random.seed(iseed)
    n = np.random.standard_normal(tants.size) * sigma_t * c_light
    return (tants + n)


@njit(**kwd)
def ADF_residuals(params, Aants, Xants, Xmax, asym_coeff=0.01):
    
    '''

    Computes amplitude residual for each antenna (i):
    residual[i] = (A_i - f_i^{ADF}(\theta,\phi,\delta\omega,A,r_xmax))**2
    where the ADF function reads:
    
    f_i = f_i(\omega_i, \eta_i, \alpha, l_i, \delta_omega, A)
        = A/l_i f_geom(\alpha, \eta_i) f_Cerenkov(\omega,\delta_\omega)
    
    where 
    
    f_geom(\alpha, \eta_i) = (1 + B \sin(\alpha))**2 \cos(\eta_i) # B is here the geomagnetic asymmetry
    f_Cerenkov(\omega_i,\delta_\omega) = 1 / (1+4{ (\tan(\omega_i)/\tan(\omega_c))**2 - 1 ) / \delta_\omega }**2 )
    
    Input parameters are: params = theta, phi, delta_omega, amplitude
    \theta, \phi define the shower direction angles, \delta_\omega the width of the Cerenkov ring, 
    A is the amplitude paramater, r_xmax is the norm of the position vector at Xmax.

    Derived parameters are: 
    \alpha, angle between the shower axis and the magnetic field
    \eta_i is the azimuthal angle of the (projection of the) antenna position in shower plane
    \omega_i is the angle between the shower axis and the vector going from Xmax to the antenna position

    '''

    theta, phi, delta_omega, amplitude = params
    nants = Aants.shape[0]
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)
    # Define shower basis vectors
    K = np.array([st*cp,st*sp,ct])
    K_plan = np.array([K[0],K[1]])
    KxB = np.cross(K,Bvec); KxB /= np.linalg.norm(KxB)
    KxKxB = np.cross(K,KxB); KxKxB /= np.linalg.norm(KxKxB)
    # Coordinate transform matrix
    mat = np.vstack((KxB,KxKxB,K))
    # 
    XmaxDist = (groundAltitude-Xmax[2])/K[2]
    # print('XmaxDist = ',XmaxDist)
    asym = asym_coeff * (1. - np.dot(K,Bvec)**2) # Azimuthal dependence, in \sin^2(\alpha)
    #
    # Make sure Xants and tants are compatible
    if (Xants.shape[0] != nants):
        print("Shapes of Aants and Xants are incompatible",Aants.shape, Xants.shape)
        return None

    # Precompute an array of Cerenkov angles to interpolate over (as in Valentin's code)
    omega_cerenkov = np.zeros(n_omega_cr+1)
    xi_table = np.arange(n_omega_cr+1)/n_omega_cr*2.*np.pi
    for i in range(n_omega_cr):
        omega_cerenkov[i] = compute_Cerenkov(xi_table[i],K,XmaxDist,Xmax,2.0e3,groundAltitude)
    # Enforce boundary condition, as numba does not like "period" keyword of np.interp
    omega_cerenkov[-1] = omega_cerenkov[0]

    # Loop on antennas
    res = np.zeros(nants)
    for i in range(nants):
        # Antenna position from Xmax
        dX = Xants[i,:]-Xmax
        # Expressed in shower frame coordinates
        dX_sp = np.dot(mat,dX)
            #
        l_ant = np.linalg.norm(dX)
        eta = np.arctan2(dX_sp[1],dX_sp[0])
        omega = np.arccos(np.dot(K,dX)/l_ant)
        # vector in the plane defined by K and dX, projected onto 
        # horizontal plane
        val_plan = np.array([dX[0]/l_ant - K[0], dX[1]/l_ant - K[1]])
        # Angle between k_plan and val_plan
        xi = np.arccos(np.dot(K_plan,val_plan)
                    /np.linalg.norm(K_plan)
                    /np.linalg.norm(val_plan))
        
        # omega_cr = compute_Cerenkov(xi,K,XmaxDist,Xmax,2.0e3,groundAltitude)
        # Interpolate to save time
        omega_cr = np.interp(xi,xi_table,omega_cerenkov)
        # omega_cr = 0.015240011539221762
        # omega_cr = np.arccos(1./RefractionIndexAtPosition(Xmax))
        # print ("omega_cr = ",omega_cr)

        # Distribution width. Here rescaled by ratio of cosines (why ?)
        width = ct / (dX[2]/l_ant) * delta_omega
        # Distribution
        adf = amplitude/l_ant / (1.+4.*( ((np.tan(omega)/np.tan(omega_cr))**2 - 1. )/width )**2)
        adf *= 1. + asym*np.cos(eta) # 
        # Chi2
        res[i]= (Aants[i]-adf)

    return(res)

@njit(**kwd)
def ADF_simulation(params, Aants, Xants, Xmax, asym_coeff=0.01, verbose=False):

    nants = Xants.shape[0]
    res = -ADF_residuals(params,np.zeros(nants), Xants, Xmax, asym_coeff=asym_coeff)
    return (res)

def function_omega(params, Aants, Xants, Xmax, asym_coeff=0.01,verbose=False):
    
    '''

    Defines Chi2 by summing *amplitude* model residuals over antennas (i):
    loss = \sum_i (A_i - f_i^{ADF}(\theta,\phi,\delta\omega,A,r_xmax))**2
    where the ADF function reads:
    
    f_i = f_i(\omega_i, \eta_i, \alpha, l_i, \delta_omega, A)
        = A/l_i f_geom(\alpha, \eta_i) f_Cerenkov(\omega,\delta_\omega)
    
    where 
    
    f_geom(\alpha, \eta_i) = (1 + B \sin(\alpha))**2 \cos(\eta_i) # B is here the geomagnetic asymmetry
    f_Cerenkov(\omega_i,\delta_\omega) = 1 / (1+4{ (\tan(\omega_i)/\tan(\omega_c))**2 - 1 ) / \delta_\omega }**2 )
    
    Input parameters are: params = theta, phi, delta_omega, amplitude
    \theta, \phi define the shower direction angles, \delta_\omega the width of the Cerenkov ring, 
    A is the amplitude paramater, r_xmax is the norm of the position vector at Xmax.

    Derived parameters are: 
    \alpha, angle between the shower axis and the magnetic field
    \eta_i is the azimuthal angle of the (projection of the) antenna position in shower plane
    \omega_i is the angle between the shower axis and the vector going from Xmax to the antenna position

    '''
    omega_array=[]

    theta, phi, delta_omega, amplitude = params
    nants = Aants.shape[0]
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)
    # Define shower basis vectors
    K = np.array([st*cp,st*sp,ct])
    K_plan = np.array([K[0],K[1]])
    KxB = np.cross(K,Bvec); KxB /= np.linalg.norm(KxB)
    KxKxB = np.cross(K,KxB); KxKxB /= np.linalg.norm(KxKxB)
    # Coordinate transform matrix
    mat = np.vstack((KxB,KxKxB,K))
    # 
    XmaxDist = (groundAltitude-Xmax[2])/K[2]
    # print('XmaxDist = ',XmaxDist)
    asym = asym_coeff * (1. - np.dot(K,Bvec)**2) # Azimuthal dependence, in \sin^2(\alpha)
    #
    # Make sure Xants and tants are compatible
    if (Xants.shape[0] != nants):
        print("Shapes of Aants and Xants are incompatible",Aants.shape, Xants.shape)
        return None

    # Precompute an array of Cerenkov angles to interpolate over (as in Valentin's code)
    omega_cerenkov = np.zeros(2*n_omega_cr+1)
    xi_table = np.arange(2*n_omega_cr+1)/n_omega_cr*np.pi
    for i in range(n_omega_cr+1):
        omega_cerenkov[i] = compute_Cerenkov(xi_table[i],K,XmaxDist,Xmax,2.0e3,groundAltitude)
    # Enforce symmetry
    omega_cerenkov[n_omega_cr+1:] = (omega_cerenkov[:n_omega_cr])[::-1]

    # Loop on antennas
    tmp = 0.
    for i in range(nants):
        # Antenna position from Xmax
        dX = Xants[i,:]-Xmax
        # Expressed in shower frame coordinates
        dX_sp = np.dot(mat,dX)
        #
        l_ant = np.linalg.norm(dX)
        eta = np.arctan2(dX_sp[1],dX_sp[0])
        omega = np.arccos(np.dot(K,dX)/l_ant)
        if Xants[i,0] > Xmax[0]:
            omega= -omega
        omega_array.append(omega)
       

        # vector in the plane defined by K and dX, projected onto 
        # horizontal plane
        val_plan = np.array([dX[0]/l_ant - K[0], dX[1]/l_ant - K[1]])
        # Angle between k_plan and val_plan
        xi = np.arccos(np.dot(K_plan,val_plan)
                       /np.linalg.norm(K_plan)
                       /np.linalg.norm(val_plan))
        
        # omega_cr = compute_Cerenkov(xi,K,XmaxDist,Xmax,2.0e3,groundAltitude)
        # Interpolate to save time
        omega_cr = np.interp(xi,xi_table,omega_cerenkov)
        # omega_cr = 0.015240011539221762
        # omega_cr = np.arccos(1./RefractionIndexAtPosition(Xmax))
        # print ("omega_cr = ",omega_cr)

        # Distribution width. Here rescaled by ratio of cosines (why ?)
        width = ct / (dX[2]/l_ant) * delta_omega
        # Distribution
        adf = amplitude/l_ant / (1.+4.*( ((np.tan(omega)/np.tan(omega_cr))**2 - 1. )/width )**2)
        adf *= 1. + asym*np.cos(eta) # 
        # Chi2
        tmp += (Aants[i]-adf)**2


    chi2 = tmp
    if (verbose):
        print ("params = ",np.rad2deg(params[:2]),params[2:]," Chi2 = ",chi2)
    return(omega_array)