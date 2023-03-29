from wavefronts import *
from recons import *
data_dir = '../Chiche/'
import timeit

def time_PWF(grad=False,hess=False,verbose=False,number=1000):

    an = antenna_set('../Chiche/coord_antennas.txt')
    co = coincidence_set('../Chiche/Rec_coinctable.txt',an)
    current_recons = 0
    args=(co.antenna_coords_array[0,:],co.peak_time_array[0,:])

    params_in = np.array([3*np.pi/4.,np.pi])
    if (grad):        
        total_time = timeit.timeit(lambda: so.minimize(PWF_loss,params_in,jac=PWF_grad,args=args,method="BFGS"),number=number)
    elif (hess):
        total_time = timeit.timeit(lambda: so.minimize(PWF_loss,params_in,jac=PWF_grad,hess=PWF_hess,args=args,method="Newton-CG"),number=number)
    else:
        total_time = timeit.timeit(lambda: so.minimize(PWF_loss,params_in,args=args,method="BFGS"),number=number)

    print ("Time to minimize loss = %.2f"%(total_time/number*1000), " (ms)")

    if verbose:
        args=(co.antenna_coords_array[0,:],co.peak_time_array[0,:],True)
    
    if (grad):
        res = so.minimize(PWF_loss,params_in,jac=PWF_grad,args=args,method='BFGS')
    elif (hess):
        res = so.minimize(PWF_loss,params_in,jac=PWF_grad,hess=PWF_hess,args=args,method='Newton-CG')
    else:
        res = so.minimize(PWF_loss,params_in,args=args,method='BFGS')
    params_out = res.x
    print ("Best fit parameters = ",*np.rad2deg(params_out))
    print ("Chi2 at best fit = ",PWF_loss(params_out,*args))
    print(res)

    return(res)

def time_SWF(grad=False,verbose=False,number=100):

    # RUN FIRST python recons.py 0 '../Chiche'
    # Read guess for theta, phi from plane wave
    fid_angles = open(data_dir+'Rec_plane_wave_recons_py.txt')
    l = fid_angles.readline().strip().split()
    theta_in,phi_in = np.float(l[2]),np.float(l[4])

    an = antenna_set('../Chiche/coord_antennas.txt')
    co = coincidence_set('../Chiche/Rec_coinctable.txt',an)
    current_recons = 0
    args=(co.antenna_coords_array[0,:],co.peak_time_array[0,:])

    # Guess parameters
    bounds = [[np.deg2rad(theta_in-1),np.deg2rad(theta_in+1)],
             [np.deg2rad(phi_in-1),np.deg2rad(phi_in+1)], 
             [-15.6e3 - 12.3e3/np.cos(np.deg2rad(theta_in)),-6.1e3 - 15.4e3/np.cos(np.deg2rad(theta_in))],
            [6.1e3 + 15.4e3/np.cos(np.deg2rad(theta_in)),0]]
    params_in = np.array(bounds).mean(axis=1)
    print ("params_in = ",params_in)

    if (grad):
        total_time = timeit.timeit(lambda: so.minimize(SWF_loss,params_in,jac=SWF_grad,args=args,method="BFGS"),number=number)
    else:
        total_time = timeit.timeit(lambda: so.minimize(SWF_loss,params_in,args=args,method="BFGS"),number=number)

    print ("Time to minimize loss = %.2f"%(total_time/number*1000), " (ms)")

    if verbose:
        args=(co.antenna_coords_array[0,:],co.peak_time_array[0,:],1,True)
    res = so.minimize(SWF_loss,params_in,args=args,method='BFGS')
    params_out = res.x
    print ("Best fit parameters = ",*np.rad2deg(params_out[:2]),*params_out[2:])
    print ("Chi2 at best fit = ",SWF_loss(params_out,*args))

    return

def time_ADF(verbose=True,number=100):

    # RUN FIRST python recons.py 1 '../Chiche'
    # Read guess for theta, phi from plane wave
    fid_angles = open(data_dir+'Rec_plane_wave_recons_py.txt')
    l = fid_angles.readline().strip().split()
    theta_in,phi_in = np.float(l[2]),np.float(l[4])

    an = antenna_set('../Chiche/coord_antennas.txt')
    co = coincidence_set('../Chiche/Rec_coinctable.txt',an)
    current_recons = 0
    # Read guess for Xmax from spherical wave
    fid_xmax = open(data_dir+'Rec_sphere_wave_recons_py.txt')
    l = fid_xmax.readline().strip().split()
    Xmax = np.array([float(l[4]),float(l[5]),float(l[6])])
    # Init parameters. Make better guess for amplitude from max of peak amplitudes
    bounds = [[np.deg2rad(theta_in-1),np.deg2rad(theta_in+1)],
             [np.deg2rad(phi_in-1),np.deg2rad(phi_in+1)],
             [0.1,3.0],
             [1e6,1e10]]
    params_in = np.array(bounds).mean(axis=1) # Central values
    ## Refine guess for amplitude, based on maximum of peak values ##
    lant = (groundAltitude-Xmax[2])/np.cos(np.deg2rad(theta_in))
    params_in[3] = co.peak_amp_array[current_recons,:].max() * lant
    print ('params_in = ',params_in)

    args=(co.peak_amp_array[current_recons,:],co.antenna_coords_array[current_recons,:],Xmax)
    total_time = timeit.timeit(lambda: so.minimize(ADF_loss,params_in,args=args,method='BFGS'),number=number)

    print ("Time to minimize loss = %.2f"%(total_time/number*1000), " (ms)")    

    if verbose:
        args=(co.peak_amp_array[current_recons,:],co.antenna_coords_array[current_recons,:],Xmax,0.01,True)

    res = so.minimize(ADF_loss,params_in,args=args,method='BFGS')
    params_out = res.x
    print ("Best fit parameters = ",*np.rad2deg(params_out[:2]),*params_out[2:])
    print ("Chi2 at best fit = ",ADF_loss(params_out,*args))


    return (res)
