import numpy as np
from numba import njit, float64, prange
from scipy.spatial.transform import Rotation as R
from scipy.optimize import fsolve, brentq
from solver import newton
from rotation import rotation
import utils as utils
import physical_parameters as phys_params
# Used for interpolation
n_omega_nr = 20

# # Physical constants
# c_light = 2.997924580e8
# R_earth = 6371007.0
# ns = 325
# kr = -0.1218
# groundAltitude = 1207.0 # 1086.0
# B_dec = 0.
# B_inc = np.pi/2. + 1.0609856522873529
# # Magnetic field direction (unit) vector
# Bvec = np.array([np.sin(B_inc)*np.cos(B_dec), np.sin(B_inc)*np.sin(B_dec), np.cos(B_inc)])

kwd = {"fastmath": {"reassoc", "contract", "arcp"}}

#@njit(**kwd)
def PWF_model(params, Xants, nr=1.0):
    '''
    Generates plane wavefront timings
    '''
    theta, phi = params
    K = utils.thetaphi_to_k(theta, phi)
    # ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp=np.sin(phi)
    # K = np.array([st*cp, st*sp, ct])
    dX = Xants - np.array([0., 0., phys_params.groundAltitude])
    tants = np.dot(dX, K) / nr / phys_params.c_light

    return (tants)


def PWF_loss(params, Xants, tants, verbose=False, nr=1.0):
    '''
    Defines Chi2 by summing model residuals
    over antenna pairs (i, j):
    loss = \sum_{i>j} ((Xants[i, :]-Xants[j, :]).K - nr(tants[i]-tants[j]))**2
    where:
    params=(theta, phi): spherical coordinates of unit shower direction vector K
    Xants are the antenna positions (shape=(nants, 3))
    tants are the antenna arrival times of the wavefront (trigger time, shape=(nants, ))
    nr is radiation speed, by default 1 since time is expressed in m.
    '''

    theta, phi = params
    nants = tants.shape[0]
    K = - utils.thetaphi_to_k(theta, phi)
    # ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)
    # K = np.array([st*cp, st*sp, ct])
    # Make sure tants and Xants are compatible
    if (Xants.shape[0] != nants):
        print("Shapes of tants and Xants are incompatible", tants.shape, Xants.shape)
        return None
    # Use numpy outer methods to build matrix X_ij = x_i -x_j
    xk = np.dot(Xants, K)
    DXK = np.subtract.outer(xk, xk)
    DT  = np.subtract.outer(tants, tants)
    chi2 = ( (DXK - nr*DT)**2 ).sum() / 2. # Sum over upper triangle, diagonal is zero because of antisymmetry of DXK, DT
    if verbose:
        print("params = ", np.rad2deg(params))
        print("Chi2 = ", chi2)
    return(chi2)

@njit(**kwd)
def PWF_alternate_loss(params, Xants, tants, verbose=False, nr=1.0):
    '''
    Defines Chi2 by summing model residuals over individual antennas, 
    after maximizing likelihood over reference time.
    '''
    nants = tants.shape[0]
    if (Xants.shape[0] != nants):
        print("Shapes of tants and Xants are incompatible", tants.shape, Xants.shape)
        return None
    # Make sure tants and Xants are compatible
    residuals = PWF_residuals(params, Xants, tants, verbose=verbose, nr=nr)
    chi2 = (residuals**2).sum()
    return chi2


def PWF_minimize_alternate_loss(Xants, tants, verbose=False, nr=1.0):
    '''
    Solves the minimization problem by using a special solution to the linear regression
    on K(\theta, \phi), with the ||K||=1 constraint. Note that this is a non-convex problem.
    This is formulated as 
    argmin_k k^T.A.k - 2 b^T.k, s.t. ||k||=1
    '''
    nants = tants.shape[0]

    # Make sure tants and Xants are compatible

    if (Xants.shape[0] != nants):
        print("Shapes of tants and Xants are incompatible", tants.shape, Xants.shape)
        return None
    # Compute A matrix (3x3) and b (3-)vector, see above
    PXT = Xants - Xants.mean(axis=0)  # P is the centering projector, XT=Xants
    A = np.dot(Xants.T, PXT)
    b = np.dot(Xants.T, tants-tants.mean(axis=0)) / nr * phys_params.c_light
    # Diagonalize A, compute projections of b onto eigenvectors
    d, W = np.linalg.eigh(A)
    beta = np.dot(b, W)
    nbeta = np.linalg.norm(beta)

    if (np.abs(beta[0]/nbeta) < 1e-14):
        if (verbose):
            print("Degenerate case")
        # Degenerate case. This will be triggered e.g. when all antennas lie in a single plane.
        mu = -d[0]
        c = np.zeros(3)
        c[1] = beta[1]/(d[1]+mu)
        c[2] = beta[2]/(d[2]+mu)
        si = np.sign(np.dot(W[:, 0], np.array([0, 0, 1.])))
        c[0] = -si*np.sqrt(1-c[1]**2-c[2]**2)  # Determined up to a sign: choose descending solution
        k_opt = np.dot(W, c)
        # k_opt[2] = -np.abs(k_opt[2]) # Descending solution

    else:
        # Assume non-degenerate case, i.e. projections on smallest eigenvalue are non zero
        # Compute \mu such that \sum_i \beta_i^2/(\lambda_i+\mu)^2 = 1, using root finding on mu
        def nc(mu):
            # Computes difference of norm of k solution to 1. Coordinates of k are \beta_i/(d_i+\mu) in W basis
            c = beta/(d+mu)
            return ((c**2).sum()-1.)
        mu_min = -d[0]+beta[0]
        mu_max = -d[0]+np.linalg.norm(beta)
        mu_opt = brentq(nc, mu_min, mu_max, maxiter=1000)
        # Compute coordinates of k in W basis, return k
        c = beta/(d+mu_opt)
        k_opt = np.dot(W, c)

    # Now get angles from k_opt coordinates
    if k_opt[2] > 1e-2:
        k_opt = k_opt-2*(k_opt@W[:, 0])*W[:, 0]

    theta_opt = np.arccos(-k_opt[2])
    phi_opt = np.arctan2(-k_opt[1], -k_opt[0])

    if phi_opt < 0:
        phi_opt += 2*np.pi

    return (np.array([theta_opt, phi_opt]))


# @njit(**kwd)
def PWF_residuals(params, Xants, tants, verbose=False, nr=1.0):

    '''
    Computes timing residuals for each antenna using plane wave model
    Note that this is defined at up to an additive constant, that when minimizing
    the loss over it, amounts to centering the residuals.
    '''
    nants = tants.shape[0]
    # Make sure tants and Xants are compatible
    if (Xants.shape[0] != nants):
        print("Shapes of tants and Xants are incompatible", tants.shape, Xants.shape)
        return None

    times = PWF_model(params, Xants, nr=nr)
    res = nr * (tants - times)
    res -= res.mean()  # Mean is projected out when maximizing likelihood over reference time t0
    return (res)


#@njit(**kwd)
def PWF_simulation(params, Xants, sigma_t=5e-9, iseed=None, nr=1.0):
    '''
    Generates plane wavefront timings, zero at shower core, with jitter noise added
    '''

    times = PWF_model(params, Xants, nr=nr)
    # Add noise
    if (iseed is not None):
        np.random.seed(iseed)
    n = np.random.standard_normal(times.size) * sigma_t
    return (times + n)


def PWF_Fisher(params, Xants, sigma_t=5e-9, nr=1.0):
    '''
    Computes the Fisher matrix for the (alternate) profile likelihood
    '''
    theta, phi = params
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp=np.sin(phi)
    K = np.array([st*cp, st*sp, ct])
    dX = Xants - np.array([0., 0., phys_params.groundAltitude])
    J = np.array([[ct*cp, ct*sp, -st], [-st*sp, st*cp, 0.]])
    rhs = np.dot(dX, J.T)  # nants x 2
    rhs = rhs - np.mean(rhs, axis=0)  # Remove mean on antennas: centering
    res = np.dot(rhs.T, rhs)
    res /= (sigma_t * phys_params.c_light)**2
    return res

### Note that these correspond to the old loss, with sums on antenna pairs

def PWF_grad(params, Xants, tants, verbose=False, nr=1.0):

    '''
    Gradient of PWF_loss, with respect to theta, phi
    '''
    theta, phi = params
    nants = tants.shape[0]
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)
    K = np.array([st*cp, st*sp, ct])

    xk = np.dot(Xants, K)
    # Use numpy outer method to build matrix X_ij = x_i - x_j
    DXK = np.subtract.outer(xk, xk)
    DT  = np.subtract.outer(tants, tants)
    RHS = DXK-nr*DT

    # Derivatives of K w.r.t. theta, phi
    dKdtheta = np.array([ct*cp, ct*sp, -st])
    dKdphi   = np.array([-st*sp, st*cp, 0.])
    xk_theta = np.dot(Xants, dKdtheta)
    xk_phi   = np.dot(Xants, dKdphi)
    # Use numpy outer method to build matrix X_ij = x_i - x_j
    DXK_THETA = np.subtract.outer(xk_theta, xk_theta)
    DXK_PHI   = np.subtract.outer(xk_phi, xk_phi)

    jac_theta = np.sum(DXK_THETA*RHS) # Factor of 2 of derivatives compensates ratio of sum to upper diag sum
    jac_phi   = np.sum(DXK_PHI*RHS)
    if verbose:
        print("Jacobian = ", jac_theta, jac_phi)
    return np.array([jac_theta, jac_phi])

def PWF_hess(params, Xants, tants, verbose=False, nr=1.0):
    '''
    Hessian of PWF_loss, with respect to theta, phi
    '''
    theta, phi = params
    nants = tants.shape[0]
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)
    K = np.array([st*cp, st*sp, ct])

    xk = np.dot(Xants, K)
    # Use numpy outer method to build matrix X_ij = x_i - x_j
    DXK = np.subtract.outer(xk, xk)
    DT  = np.subtract.outer(tants, tants)
    RHS = DXK-nr*DT

    # Derivatives of K w.r.t. theta, phi
    dK_dtheta = np.array([ct*cp, ct*sp, -st])
    dK_dphi   = np.array([-st*sp, st*cp, 0.])
    d2K_dtheta= np.array([-st*cp, -st*sp, -ct])
    d2K_dphi  = np.array([-st*cp, -st*sp, 0.])
    d2K_dtheta_dphi = np.array([-ct*sp, ct*cp, 0.]) 

    xk_theta = np.dot(Xants, dK_dtheta)
    xk_phi   = np.dot(Xants, dK_dphi)
    xk2_theta = np.dot(Xants, d2K_dtheta)
    xk2_phi   = np.dot(Xants, d2K_dphi)
    xk2_theta_phi = np.dot(Xants, d2K_dtheta_dphi)

    #Use numpy outer method to buid matrix X_ij = x_i - x_j
    DXK_THETA = np.subtract.outer(xk_theta, xk_theta)
    DXK_PHI   = np.subtract.outer(xk_phi, xk_phi)
    DXK2_THETA = np.subtract.outer(xk2_theta, xk2_theta)
    DXK2_PHI   = np.subtract.outer(xk2_phi, xk2_phi)
    DXK2_THETA_PHI = np.subtract.outer(xk2_theta_phi, xk2_theta_phi)

    hess_theta2 = np.sum(DXK2_THETA*RHS + DXK_THETA**2)
    hess_phi2   = np.sum(DXK2_PHI*RHS + DXK_PHI**2)
    hess_theta_phi = np.sum(DXK2_THETA_PHI*RHS + DXK_THETA*DXK_PHI)

    return (np.array([[hess_theta2, hess_theta_phi], [hess_theta_phi, hess_phi2]]))