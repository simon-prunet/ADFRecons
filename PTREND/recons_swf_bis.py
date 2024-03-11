import numpy as np


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from utils import c, R2D, cart2sph, sph2cart
# from visu_distrib import scatter_sphere, plotting_mesh, create_sphericalmesh
from scipy.stats import multivariate_normal as norm
import time
R2D = np.pi/180

def _make_XY(Xants,tants, cr=1.):
    """X of shape N_event*N_antenna*4"""
    y = cr**2*tants**2 - np.einsum('ij,ij -> i', Xants, Xants)
    X = 2*np.concatenate((-Xants, cr*tants[:,None]),
                          axis=1)
    return X,y

def _make_XY1col(Xants,tants, cr=1.):
    """X of shape N_event*N_antenna*4"""
    y = cr**2*tants**2 - np.einsum('ij,ij -> i', Xants, Xants)
    X = np.concatenate((-2*Xants, 2*cr*tants[:,None], np.ones((len(tants), 1))),
                          axis=1)
    return X,y

def linear(Xants,tants,cr, iterate=False):
    """
    Applying reconstruction method from 
    Chan, Y., Ho, K., 1994. A simple and efficient estimator for hyperbolic location
    Need to test if iterate brings something more.
    """
    
    X, Y = _make_XY1col(Xants, tants, cr)
    X = np.matrix(X)
    Y = np.matrix(Y)
    #Covariance matrix
    w1 = (X.T*X)**(-1) *X.T*Y.T
    w1 = np.array(w1)[:,0]
    res = w1[:4]
    b_pred = w1[4]
    return res, b_pred

def Linear2(Xants,tants, cr=1.):
    n_ants = len(tants)
    X,y = _make_XY(Xants, tants, cr)
    X_centered = np.matrix(X-X.mean(axis=0)[None,:])
    y_centered = np.matrix(y - y.mean())
    XTX_1 = (X_centered.T*X_centered)**(-1)
    res = XTX_1*X_centered.T*y_centered.T
    res = np.array(res)[:,0]
    b_pred = y.mean() - X.mean(axis=0).dot(res)
    return res, b_pred
Linear = Linear2
#To test: All these methods return [xs,ys,zs], 
#to get ts, replace all the [:3] in the return lines by [:4] 
#TDOA1 does not output ts 

def approx_A(Xants,tants, cr=1.):
    return Linear(Xants,tants, cr)[0][:3]

def approx_B(Xants,tants, cr=1.):
    res, b_pred = Linear(Xants,tants, cr)
    z_0_2 = np.sqrt(b_pred + res[3]**2 - res[0]**2 - res[1]**2)
    res[2] = np.sign(res[2])*np.abs(z_0_2)
    return res[:4]      #If time needed, replace 3 by 4

def approx_C(Xants,tants, cr=1.):
    C = Xants.T@Xants
    d, R = np.linalg.eigh(C)
    R = R[:,::-1]
    P_rot = (R.T@Xants.T).T
    res = approx_B(P_rot,tants, cr)

    if len(res)==4:
        res = np.append(R@res[:3].flatten(), res[-1])
        return res
    
    return (R@res[:4]).flatten()

def TDOA_ant1(Xants,tants,cr, iterate=False):
    """
    Applying reconstruction method from 
    Chan, Y., Ho, K., 1994. A simple and efficient estimator for hyperbolic location
    Need to test if iterate brings something more.
    """
    n_ants = len(Xants)

    delta_t_i = cr*(tants[1:]-tants[0])
    Pn2 = np.einsum('ij,ij->i', Xants,Xants)

    X = np.matrix(2*np.concatenate((Xants[1:,:] - Xants[[0],:], delta_t_i[:,None]), axis=1))
    Y = np.matrix(Pn2[1:] - Pn2[0]-delta_t_i**2)

    #Covariance matrix
    Q = np.matrix(np.identity(n_ants-1 ) + 1, dtype=np.float32)
    Q_1 = Q**(-1)
    w1 = (X.T*Q_1*X)**(-1) *X.T*Q_1*Y.T
    w1 = np.array(w1)[:,0]
    if iterate:
        #Do it again with a Q that is more precise
        Q = np.matrix(np.diag(np.linalg.norm(Xants[1:]-w1[:3], axis=1) ))*Q
        Q_1 = Q**(-1)
        w1 = (X.T*Q_1*X)**(-1) * X.T*Q_1*Y.T
        w1 = np.array(w1)[:,0]


    Cov1 = (X.T*Q_1*X)**(-1)

    #APPLYING THE CONSTAINT
    D = np.concatenate( (w1[:-1]-Xants[0], w1[[-1]]) )
    D2 = np.matrix(D**2)

    G = np.matrix([
        [1,0,0],
        [0,1,0],
        [0,0,1],
        [1,1,1]
    ])
    
    B = np.matrix(np.diag(D))
    Phi = B*Cov1*B
    Phi_1 = Phi**(-1)
    w2 = (G.T*Phi_1*G)**(-1)*G.T*Phi_1*D2.T
    w2 = np.array(w2)[:,0]
    pos_opt = np.sign((w1[:3]-Xants[0]))*np.sqrt(w2)+Xants[0]
    return list(pos_opt.flatten()) + [0]

## Not working, useless for now
def TDOA_antN(Xants,tants,cr, iterate=False):
    """
    Applying reconstruction method from 
    Chan, Y., Ho, K., 1994. A simple and efficient estimator for hyperbolic location
    Need to test if iterate brings something more.
    """
    
    X, Y = _make_XY1col(Xants, tants, cr)
    X = np.matrix(X)
    Y = np.matrix(Y)
    #Covariance matrix
    Q = np.matrix(np.identity(len(X)))
    Q_1 = Q
    w1 = (X.T*X)**(-1) *X.T*Y.T
    w1 = np.array(w1)[:,0]
    return w1[:3]
    if iterate:
        #Do it again with a Q that is more precise
        print(w1.shape)
        Q = np.diag(np.linalg.norm(Xants[:]-w1[1:4], axis=1) )
        Q = np.matrix(Q)
        Q_1 = Q**(-1)
        w1 = (X.T*Q_1*X)**(-1) * X.T*Q_1*Y.T
        w1 = np.array(w1)[:,0]


    Cov1 = (X.T*X)**(-1)

    #APPLYING THE CONSTRAINT
    D = np.matrix(np.concatenate( (w1[:-1]**2, w1[[-1]]) ))

    G = np.matrix([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
        [-1,1,1,1]
    ])
    
    B = np.matrix(np.diag( list(2*w1[:-1])+[1] ))
    C = B*Cov1*B
    C_1 = C**(-1)

    w2 = (G.T*C_1*G)**(-1)*G.T*C_1*D.T
    w2 = np.array(w2)[:,0]

    pos_opt = np.sign(w1[1:4])*np.sqrt(w2[1:])
    return pos_opt
