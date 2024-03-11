import numpy as np

def PWF_methodA(Xants, tants, verbose=False, cr=1.0):
    '''
    Here: 
    Xants N x 3 
    tants N
    cr: speed of light
    '''
    Pn = (Xants-Xants.mean(axis=0))
    pseudoinverse = np.linalg.inv(Pn.T@Pn)
    M = pseudoinverse @ Pn.T
    k_lin = cr*M@(tants-tants.mean())

    k_opt = k_lin/np.linalg.norm(k_lin)

    # Now get angles from k_opt coordinates
    theta_opt = np.arccos(k_opt[2])
    phi_opt = np.arctan2(k_opt[1],k_opt[0])
    #return theta_opt, phi_opt
    return(np.array([theta_opt,phi_opt]))


def PWF_methodB(Xants, tants, verbose=False, cr=1.0):
    '''
    Here: 
    Xants N x 3 
    tants N
    cr: speed of light
    '''
    Pn = (Xants[:,:2]-Xants[:,:2].mean(axis=0))/cr
    pseudoinverse = np.linalg.inv(Pn.T@Pn)
    M = pseudoinverse @ Pn.T
    k_lin = M@(tants-tants.mean())

    n2 = np.linalg.norm(k_lin)
    k_opt = [*(k_lin/max(1,n2)), - np.sqrt(1 - min(1, n2**2))]
    print('!!!!', k_opt)

    # Now get angles from k_opt coordinates
    theta_opt = np.arccos(k_opt[2])
    phi_opt = np.arctan2(k_opt[1],k_opt[0])
    #return theta_opt, phi_opt
    return(np.array([theta_opt,phi_opt]))


def PWF_methodC(Xants, tants, verbose=False, cr=1.0):
    '''
    Here: 
    Xants N x 3 
    tants N
    cr: speed of light
    '''
    Pn = (Xants[:,:]-Xants[:,:].mean(axis=0))
    pseudoinverse = np.linalg.inv(Pn.T@Pn)
    M = pseudoinverse @ Pn.T
    k_lin = cr*M@(tants-tants.mean())

    #Q = np.linalg.eigh(pseudoinverse)[1][:,::-1]
    #k_lin_rot = np.einsum("kj,k->j", Q, k_lin)

    Q = np.linalg.eigh(pseudoinverse)[1]
    k_lin_rot = np.einsum("kj,k->j", Q, k_lin)[:2]
    
    sign = np.sign(k_lin_rot[-1])
    n2 = np.linalg.norm(k_lin_rot)
    print('???',n2)
    k_opt_rot = [*(k_lin_rot/max(1,n2)), sign*np.sqrt(1 - min(1, n2**2))]       #Need to verify sign
    print('!!!!!!!',k_opt_rot)
    print(k_opt_rot)
    # k_opt_rot[2] = -k_opt_rot[2]
    # k_opt_rot[2] = -np.abs(k_opt_rot[2])
    # k_opt_rot[2] = np.abs(k_opt_rot[2])

    k_opt = np.einsum("jk,k->j", Q, k_opt_rot)       #Maybe need to change sign here also
    k_opt[2] = -np.abs(k_opt[2])
    # k_opt[2] = np.abs(k_opt[2])

    # Now get angles from k_opt coordinates
    theta_opt = np.arccos(k_opt[2])
    phi_opt = np.arctan2(k_opt[1],k_opt[0])
    #return theta_opt, phi_opt
    return(np.array([theta_opt,phi_opt]))
