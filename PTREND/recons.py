import numpy as np
from wavefronts import *
from recons_PWF import *
from recons_swf_bis import *
import sys
import os
import scipy.optimize as so
import numdifftools as nd
from emcee import EnsembleSampler
import time
import signal


c_light = 2.997924580e8

#output_directory = '/sps/grand/mguelfand/DC2/output_recons_ADF/'
#output_directory = '/Users/mguelfan/Documents/GRAND/ADF_DC2/output_recons_ADF/'
#output_directory = '/Users/mguelfan/Documents/GRAND/ADF_DC2/output_recons_linear/'
#output_directory = '/Users/mguelfan/Documents/GRAND/ADF_DC2/output_recons_linear/conservative/'
#output_directory = '/Users/mguelfan/Documents/GRAND/ADF_DC2/output_recons_test/aggressive/'
#output_directory = '/Users/mguelfan/Documents/GRAND/ADF_DC2/transients_GP13/reconstruction_2/'
#output_directory = '/Users/mguelfan/Documents/GRAND/ADF_DC2/transients_GP13/february2024/'
output_directory = '/Users/mguelfan/Documents/GRAND/ADF_DC2/output_recons_simusgp300_Dunhuang2022/complete_pipeline_aggressive/'

def handler(signum, frame):
    raise TimeoutError("Le temps limite est dépassé.")
signal.signal(signal.SIGALRM, handler)


class antenna_set:

    def __init__(self, position_file):

        if (not os.path.exists(position_file)):
            print("Antenna coordinates file %s does not exist"%position_file)
            return
        print(" Reading antenna positions from file %s"%position_file)
        self.position_input_file = position_file
        self.indices = np.loadtxt(self.position_input_file, usecols=(0,), dtype=int)
        self.coordinates = np.loadtxt(self.position_input_file, usecols = (1,2,3))
        self.init_ant = np.min(self.indices)
        self.nants = np.size(self.indices)
        return

class coincidence_set:


    def __init__(self, coinc_table_file, antenna_set_instance):

        if (not os.path.exists(coinc_table_file)):
            print("Coincidence table file %s does not exist"%coinc_table_file)
            return

        self.coinc_table_file = coinc_table_file
        # This assumes an antenna_set instance has been created first
        if (not isinstance(antenna_set_instance,antenna_set)):
            print("Usage: co = coincidence_set(coinc_table_file,antenna_set_instance)")
            print("where coinc_table_file is a coincidence table file, and")
            print("antenna_set_instance is an instance of the class antenna_set")
            return
        self.ant_set = antenna_set_instance
        print(" Reading coincidence(s): index, peak time, peak amplitude from file %s"%self.coinc_table_file)
        tmp = np.loadtxt(self.coinc_table_file,dtype='int',usecols=(0,1))
        antenna_index_array = tmp[:,0]
        coinc_index_array   = tmp[:,1]
        tmp2 = np.loadtxt(self.coinc_table_file,usecols=(2,3)) # floats
        peak_time_array = tmp2[:,0]*c_light
        peak_amp_array  = tmp2[:,1]
        coinc_indices = np.unique(coinc_index_array)
        ncoincs = len(coinc_indices)

        # Store number of antennas per coincidence event
        self.nants = np.zeros(ncoincs,dtype='int')

        # Filter out coincidences with small number of antennas
        # A bit of complexity here since the number of antennas involved per coincidence event
        # will vary, so we need to keep track of the number of antennas per event.
        self.ncoincs = 0
        self.nantsmax = 0
        for index in coinc_indices:
            current_mask = (coinc_index_array==index)
            current_length = np.sum(current_mask)
            if current_length>=3:
                self.nants[self.ncoincs] = current_length
                self.nantsmax = np.maximum(self.nantsmax, current_length)
                self.ncoincs += 1

        #print(self.nants,self.ncoincs)
        # Now create the structure and populate it
        self.antenna_index_array = np.zeros((self.ncoincs,self.nantsmax),dtype='int')
        self.antenna_coords_array= np.zeros((self.ncoincs,self.nantsmax,3))
        self.coinc_index_array   = np.zeros((self.ncoincs,self.nantsmax),dtype='int')
        self.peak_time_array     = np.zeros((self.ncoincs,self.nantsmax))
        self.peak_amp_array      = np.zeros((self.ncoincs,self.nantsmax))

        # Filter and read
        current_coinc = 0
        for index in coinc_indices:
            mask = (coinc_index_array==index)
            # print (mask)
            current_length = np.sum(mask)
            if current_length>=3:
                # Next line assumes that the antenna coordinate files gives all antennas in order, starting from antenna number=init_ant
                # This will be needed to get antenna coordinates per coincidence event, from the full list in antenna_set
                self.antenna_index_array[current_coinc,:self.nants[current_coinc]] = antenna_index_array[mask]-self.ant_set.init_ant
                self.antenna_coords_array[current_coinc,:self.nants[current_coinc],:] = self.ant_set.coordinates[antenna_index_array[mask]]
                #print(len(self.antenna_coords_array[current_coinc,:self.nants[current_coinc],:]))
                # Now read coincidence index (constant within the same coincidence event !), peak time and peak amplitudes per involved antennas.
                self.coinc_index_array[current_coinc,:self.nants[current_coinc]] = coinc_index_array[mask]
                self.peak_time_array[current_coinc,:self.nants[current_coinc]] = peak_time_array[mask]
                #print(len(self.peak_time_array[current_coinc,:self.nants[current_coinc]]))
                #self.peak_time_array[current_coinc,:self.nants[current_coinc]] -= np.min(self.peak_time_array[current_coinc,:self.nants[current_coinc]])
                self.peak_amp_array[current_coinc,:self.nants[current_coinc]] = peak_amp_array[mask]
                current_coinc += 1
        return

class setup:

    def __init__(self, data_dir,recons_type, compute_errors=False):

        self.recons_type = recons_type
        self.data_dir    = data_dir
        self.compute_errors = compute_errors
        if (self.recons_type<0 or self.recons_type>2):
            print("Choose reconstruction type values in :")
            print("0: plane wave reconstruction")
            print("1: spherical wave reconstruction")
            print("2: ADF model reconstruction")
            print("Other values not supported.")
            return
        if (not os.path.exists(self.data_dir)):
            print("Data directory %s does not seem to exist."%self.data_dir)
            return

        # Prepare output files
        if (self.recons_type==0):
            self.outfile = self.data_dir+'/Rec_plane_wave_recons.txt'
        elif (self.recons_type==1):
            self.outfile = self.data_dir+'/Rec_sphere_wave_recons.txt'
        elif (self.recons_type==2):
            self.outfile = self.data_dir+'/Rec_adf_recons.txt'
            self.outfile_res = self.data_dir+'/Rec_adf_amplitude_residuals_arccos_refractiveindex.txt'

        if os.path.exists(self.outfile):
            # Remove previous files
            os.remove(self.outfile)


        # Prepare input files, depending on reconstruction type
        if (self.recons_type==1 or self.recons_type==2):
            self.input_angles_file = self.data_dir+'/Rec_plane_wave_recons.txt'
        if (self.recons_type==2):
            self.input_xmax_file = self.data_dir+'/Rec_sphere_wave_recons.txt'


    def write_angles(self,outfile,coinc,nants,angles,errors,chi2, time):

        fid = open(outfile,'a')
        fid.write("%ld %3.0d %12.5le %12.5le %12.5le %12.8le %12.5le %12.5le %12.5le \n"%(coinc,nants,angles[0],errors[0],angles[1],errors[1],chi2,np.nan, time))
        fid.close()

    def write_xmax(self,outfile,coinc,nants,params,chi2,time):
        fid = open(outfile,'a')
        theta,phi,r_xmax,t_s = params
        st=np.sin(theta); ct=np.cos(theta); sp=np.sin(phi); cp=np.cos(phi); K = [st*cp,st*sp,ct]
        fid.write("%ld %3.0d %12.5le %12.5le %12.5le %12.5le %12.5le %12.5le %12.5le %12.5le \n"%(coinc,nants,chi2,np.nan,-r_xmax*K[0],-r_xmax*K[1],groundAltitude-r_xmax*K[2],t_s, r_xmax, time))
        fid.close()

    def write_xmax_false(self,outfile,coinc):
        fid = open(outfile,'a')
        fid.write("%ld %3.0f %12.0f %12.0f %12.0f %12.0f %12.0f %12.0f %12.0f \n"%(coinc, -1 ,-1 , -1, -1, -1, -1, -1, -1))
        fid.close()

    def write_xmax_SWF_analitycal(self,outfile,coinc,nants,params,chi2,time):
        fid = open(outfile,'a')
        x_recons, y_recons, z_recons, t_s = params
        #st=np.sin(theta); ct=np.cos(theta); sp=np.sin(phi); cp=np.cos(phi); K = [st*cp,st*sp,ct]
        fid.write("%ld %3.0d %12.5le %12.5le %12.5le %12.5le %12.5le %12.5le %12.5le \n"%(coinc,nants,chi2,np.nan, x_recons, y_recons, z_recons,t_s, time))
        fid.close()


    def write_adf(self,outfile,coinc,nants,params,errors,chi2, time):
        fid = open(outfile,'a')
        theta,phi,delta_omega,amplitude = params
        theta_err, phi_err, delta_omega_err, amplitude_err = errors
        format_string = "%ld %3.0d "+"%12.5le "*9+"\n"
        fid.write(format_string%(coinc,nants,np.rad2deg(theta),np.rad2deg(theta_err),
            np.rad2deg(phi),np.rad2deg(phi_err),chi2, np.nan,delta_omega,amplitude, time))
        fid.close()

    def write_adf_false(self, outfile, coinc):
        fid = open(outfile,'a')
        format_string = "%ld %3.0f "+"%12.0f "*9+"\n"
        fid.write(format_string%(coinc, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1))
        fid.close()


    def write_amplitude_residuals(self, outfile_res, coinc, nants, amplitude_simu, residuals, amplitude_recons, eta_recons, omega_recons, coord):
        fid = open(outfile_res,'a')
        coinc = [coinc] * len(residuals)
        nants = [nants] * len(residuals)
        amplitude_simu = [amplitude_simu] * len(residuals)
        #fid.write(f"{coinc}\t{nants}\t{coord}\t{residuals}\n")
        for coinci, n, s, r, a, e, o, c in zip(coinc, nants, amplitude_simu, residuals, amplitude_recons, eta_recons, omega_recons, coord):
            fid.write(f"{coinci}\t{n}\t{s}\t{r}\t{a}\t{e}\t{o}\t{c[0]}\t{c[1]}\t{c[2]}\n")
        fid.close()

def main():

    if (len(sys.argv) != 3):
        print ("Usage: python recons.py <recons_type> <data_dir> ")
        print ("recons_type = 0 (plane wave), 1 (spherical wave), 2 (ADF)")
        print ("data_dir is the directory containing the coincidence files")
        sys.exit(1)

    recons_type = int(sys.argv[1])
    data_dir = sys.argv[2]

    print('recons_type = ',recons_type)
    print('data_dir = ',data_dir)

    # Read antennas indices and coordinates
    an = antenna_set(data_dir+'/coord_antennas.txt')
    # Read coincidences (antenna index, coincidence index, peak time, peak amplitude)
    # Routine only keep valid number of antennas (>3)
    co = coincidence_set(data_dir+'/Rec_coinctable.txt',an)
    #print("Number of coincidences = ",co.ncoincs)
    # Initialize reconstruction
    st = setup(data_dir,recons_type)

    if (st.recons_type==0):
        # PWF model. We do not assume any prior analysis was done.
            for current_recons in range(co.ncoincs):
                begining_time = time.time()
                bounds = ((np.pi/2+1e-7,np.pi),(0,2*np.pi))
                #bounds = ((1e-7,np.pi),(0,2*np.pi))
                params_in = np.array(bounds).mean(axis=1)
                # args=(co.antenna_coords_array[current_recons,:],co.peak_time_array[current_recons,:],True)
                args=(co.antenna_coords_array[current_recons,: co.nants[current_recons]],co.peak_time_array[current_recons,:co.nants[current_recons]])
                #args = (co.antenna_coords_array[current_recons,:], co.peak_time_array[current_recons,:])
               
                #test different minimization methods (gradient descent, MCMC, analytical):
                #res = so.minimize(PWF_loss,params_in,args=args,method='BFGS')
                #res = so.minimize(PWF_loss,params_in, jac=PWF_grad, args=args, method='L-BFGS-B', bounds=bounds)
                #res = so.minimize(PWF_loss, params_in, args=args, bounds=bounds, method='Nelder-Mead')
                #params_out = res.x
                #print(res.success)
                #res = MCMC_minimizer(logprob, args=args)
                res = PWF_minimize_alternate_loss(co.antenna_coords_array[current_recons,: co.nants[current_recons]],co.peak_time_array[current_recons,:co.nants[current_recons]])
                #res = MCMC_minimizer(logprob_alternate, args=args)
                #res = so.minimize(PWF_loss,res.x,args=(co.antenna_coords_array[current_recons,:],co.peak_time_array[current_recons,:],1,True),method='L-BFGS-B')    
                #res = PWF_methodC(co.antenna_coords_array[current_recons,: co.nants[current_recons]],co.peak_time_array[current_recons,:co.nants[current_recons]])
                params_out = res

                if (st.compute_errors):
                    args=(co.antenna_coords_array[current_recons,:],co.peak_time_array[current_recons,:])
                    hess = nd.Hessian(PWF_loss) (params_out,*args)
                    errors = np.sqrt(np.diag(np.linalg.inv(hess)))
                else:
                    errors = np.array([np.nan]*2)
                print ("Best fit parameters = ",np.rad2deg(params_out))
                ## Errors computation needs work: errors are coming both from noise on amplitude and time measurements
                if (st.compute_errors):
                    print ("Errors on parameters (from Hessian) = ",np.rad2deg(errors))
                #print ("Chi2 at best fit = ",PWF_loss(params_out,*args))
                #print ("Chi2 at best fit \pm errors = ",PWF_loss(params_out+errors,*args),PWF_loss(params_out-errors,*args))
                end_time = time.time()
                plane_time = end_time - begining_time
                # Write down results to file
                st.write_angles(st.outfile,co.coinc_index_array[current_recons,0],co.nants[current_recons],
                    np.rad2deg(params_out),np.rad2deg(errors),PWF_loss(params_out,*args), plane_time)

    if (st.recons_type==1):
        # SWF model. We assume that PWF reconstrution was run first. Check if corresponding result file exists.
        if not os.path.exists(st.input_angles_file):
            print("SWF reconstruction was requested, while input angles file %s does not exists."%st.input_angles_file)
            return
        fid_input_angles = open(st.input_angles_file,'r')
        #fid_input_angles_bis = open('/Users/mguelfan/Documents/GRAND/ADF_DC2/output_recons_ADF/input_simus.txt', 'r')
        i=0
        with open(f'{output_directory}input_simus.txt', 'r') as file:
            lines = file.readlines()
        with open(f'{output_directory}Rec_plane_wave_recons.txt', 'r') as filebis:
            linebis = filebis.readlines()
        with open(f'{output_directory}input_simus_bis.txt', 'w') as file_input:
            file_input.write('')
        with open(f'{output_directory}Rec_plane_wave_reconsbis.txt', 'w') as file_recons:
            file_recons.write('')
        #plane_parameters = np.loadtxt('/Users/mguelfan/Documents/GRAND/ADF_DC2/output_recons_ADF/Rec_plane_wave_recons.txt')
        #theta_reconstructed = plane_parameters[:,2]
        #print('theta_recons', theta_reconstructed)
        j=0
        timeout = 200
        for current_recons in range(co.ncoincs):
                signal.alarm(timeout)
                begining_time = time.time()
                try:
                    # Read angles obtained with PWF reconstruction
                    l = fid_input_angles.readline().strip().split()
                    if l != 'nan':
                        theta_in = float(l[2])
                    #if theta_in <= np.rad2deg(0) or theta_in >= np.rad2deg(np.pi) or theta_in == -1:
                    if theta_in <= np.rad2deg(np.pi/2) or theta_in >= np.rad2deg(np.pi) or theta_in == -1:
                        params_out = [-1, -1, -1, -1]
                        with open(f'{output_directory}Rec_plane_wave_reconsbis.txt', 'a') as file:
                            file.write(f'{co.coinc_index_array[current_recons,0]} -1 -1 -1 -1 -1 -1 -1 -1 \n')
                        with open(f'{output_directory}input_simus_bis.txt', 'a') as filebis:
                            filebis.write(f'{co.coinc_index_array[current_recons,0]} -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 \n')
                        #st.write_xmax(st.outfile,co.coinc_index_array[current_recons,0], -1, params_out, -1, -1)
                        st.write_xmax_false(st.outfile,co.coinc_index_array[current_recons,0])
                    else:
                        phi_in   = float(l[4])
                        bounds = [[np.deg2rad(theta_in-1),np.deg2rad(theta_in+1)],
                                [np.deg2rad(phi_in-1),np.deg2rad(phi_in+1)], 
                                [0, 10e5],
                               #[-15.6e3 - 12.3e3/np.cos(np.deg2rad(theta_in)),-6.1e3 - 15.4e3/np.cos(np.deg2rad(theta_in))],
                                [-10e5,0]]
                                #[6.1e3 + 15.4e3/np.cos(np.deg2rad(theta_in)),0]]
                                
                        params_in = np.array(bounds).mean(axis=1)
                        args=(co.antenna_coords_array[current_recons,:co.nants[current_recons]],co.peak_time_array[current_recons,:co.nants[current_recons]],False)
                        # args=(co.antenna_coords_array[current_recons,:],co.peak_time_array[current_recons,:])

                        # Test value of gradient, compare to finite difference estimate
                        # print(nd.Gradient(SWF_loss,order=4)(params_in,*args))
                        # print(SWF_grad(params_in,*args))
                        method = 'L-BFGS-B'
                        print('Minimize using %s'%method)
                        res = so.minimize(SWF_loss,params_in,args=args,bounds=bounds,method=method,options={'ftol':1e-13})
                        print('xxxxx')
                        res = so.minimize(SWF_loss,res.x,args=args,bounds=bounds,method='Nelder-Mead',options={'maxiter':400})
                        #res = so.minimize(SWF_loss,params_in,jac=SWF_grad,args=args,method='BFGS')
                        params_out = res.x

                        #analytical SWF
                        #res = TDOA_ant1(co.antenna_coords_array[current_recons,: co.nants[current_recons]],co.peak_time_array[current_recons,:co.nants[current_recons]], cr=1.0)
                        #res = approx_B(co.antenna_coords_array[current_recons,: co.nants[current_recons]],co.peak_time_array[current_recons,:co.nants[current_recons]], cr=1.0)
                        #res = approx_C(co.antenna_coords_array[current_recons,: co.nants[current_recons]],co.peak_time_array[current_recons,:co.nants[current_recons]], cr=1.0)
                        #params_out = res
                        #X0 = co.antenna_coords_array[current_recons,: co.nants[current_recons]][0]
                        #print(X0)
                        #n_average = ZHSEffectiveRefractionIndex(X0,np.array(res[0:3]))
                        #print('params_out', params_out)
                        #print(n_average)
                        #res = TDOA_ant1(co.antenna_coords_array[current_recons,: co.nants[current_recons]],co.peak_time_array[current_recons,:co.nants[current_recons]], cr= 1/n_average)
                        #params_out = res
                        # Compute errors with numerical estimate of Hessian matrix, inversion and sqrt of diagonal terms
                        if (st.compute_errors):
                            args=(co.antenna_coords_array[current_recons,:],co.peak_time_array[current_recons,:])
                            hess = nd.Hessian(SWF_loss)(params_out,*args)
                            errors = np.sqrt(np.diag(np.linalg.inv(hessian)))
                        else:
                            errors = np.array([np.nan]*2)      

                        print ("Best fit parameters = ",*np.rad2deg(params_out[:2]),*params_out[2:])
                        #print ("Chi2 at best fit = ",SWF_loss(params_out,*args,False))
            
                        # Compute gradient with SWF_grad and compare to finite difference estimate
                        # print(nd.Gradient(SWF_loss)(params_out,*args))
                        # print(SWF_grad(params_out,*args))

                        #print ("Chi2 at best fit \pm errors = ",SWF_loss(params_out+errors,*args),SWF_loss(params_out-errors,*args))
                        # Write down results to file 
                        
                        filebis = open(f'{output_directory}Rec_plane_wave_reconsbis.txt', 'a') 
                        linebis_copy = linebis[current_recons]
                        filebis.writelines(linebis_copy)
                        filebis.close()
                        file = open(f'{output_directory}input_simus_bis.txt', 'a')
                        lines_copy = lines[current_recons]
                        file.writelines(lines_copy)
                        file.close()
                        
                        end_time = time.time()
                        sphere_time = end_time - begining_time
                        #select output file either for descent gradient or SWF analytical
                        st.write_xmax(st.outfile,co.coinc_index_array[current_recons,0],co.nants[current_recons],params_out,SWF_loss(params_out,*args), sphere_time)
                        #st.write_xmax_SWF_analitycal(st.outfile,co.coinc_index_array[current_recons,0],co.nants[current_recons],params_out, sphere_time, sphere_time)
                        i+=1
                        #print(i)

                except TimeoutError as e:
                    print("error !!!!!!!!!!", e) 
                    params_out = [-1, -1, -1, -1]
                    with open(f'{output_directory}Rec_plane_wave_reconsbis.txt', 'a') as file:
                        file.write(f'{co.coinc_index_array[current_recons,0]} -1 -1 -1 -1 -1 -1 -1 -1 \n')
                    with open(f'{output_directory}input_simus_bis.txt', 'a') as filebis:
                        filebis.write(f'{co.coinc_index_array[current_recons,0]} -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 \n')
                    #st.write_xmax(st.outfile,co.coinc_index_array[current_recons,0], -1, params_out, -1, -1)  
                    st.write_xmax_false(st.outfile,co.coinc_index_array[current_recons,0])             
                    continue
                finally:
                    signal.alarm(0) 


    if (st.recons_type==2):
        # ADF model. We assume that PWF and SWF reconstructions were run first. Check if corresponding result files exist.
        if not os.path.exists(st.input_angles_file):
            print("ADF reconstruction was requested, while input input angles file %s dos not exists."%st.input_angles_file)
            return
        if not os.path.exists(st.input_xmax_file):
            print ("ADF reconstruction was requested, while input xmax file %s does not exists."%st.input_xmax_file)
            return
        #fid_input_angles = open(st.input_angles_file,"r")
        #fid_input_xmax   = open(st.input_xmax_file,"r")
        fid_input_angles = open(f'{output_directory}Rec_plane_wave_reconsbis.txt',"r")
        fid_input_xmax   = open(f'{output_directory}Rec_sphere_wave_recons.txt',"r")
        fid_input_xmaxsimu = open(f'{output_directory}input_simus_bis.txt',"r")
        #co.ncoins == len(fid_input_angles)
        for current_recons in range(co.ncoincs):
        #for current_recons in range(len(length_data)):
            try:
                begining_time = time.time()
            #for current_recons in range(len(fid_input_angles)):
                # Read angles obtained with PWF reconstruction
                l = fid_input_angles.readline().strip().split()
                theta_in = float(l[2])
                phi_in   = float(l[4])
                l = fid_input_xmax.readline().strip().split()
                l_simu = fid_input_xmaxsimu.readline().strip().split()
                #here, reconstructed Xmax
                Xmax = np.array([float(l[4]),float(l[5]),float(l[6])])
                #to have fixed Xmax from the simulations
                #Xmax = np.array([float(l_simu[7]),float(l_simu[8]),float(l_simu[9])])
                bounds = [[np.deg2rad(theta_in-1),np.deg2rad(theta_in+1)],
                            [np.deg2rad(phi_in-1),np.deg2rad(phi_in+1)],
                            [0.1,3.0],
                            [1e6,1e10]]
                params_in = np.array(bounds).mean(axis=1)
                #print(params_in[2])
                if theta_in  == -1:
                    st.write_adf_false(st.outfile, co.coinc_index_array[current_recons,0])
                    #st.write_adf(st.outfile, co.coinc_index_array[current_recons,0], -1, [-1, -1, -1, -1], [-1, -1, -1, -1], -1, -1)
                else:
                    lant = (groundAltitude-Xmax[2])/np.cos(np.deg2rad(theta_in))
                    #print(lant)
                    params_in[3] = co.peak_amp_array[current_recons,:co.nants[current_recons]].max() * lant
                    #print ('amp_guess = ',params_in[3])
                    ###################
                    args = (co.peak_amp_array[current_recons,:co.nants[current_recons]],co.antenna_coords_array[current_recons,:co.nants[current_recons]],Xmax, 0.01, False)
                    res = so.minimize(ADF_loss, params_in,args=(co.peak_amp_array[current_recons,:co.nants[current_recons]],co.antenna_coords_array[current_recons,:co.nants[current_recons]],Xmax),
                                       method='L-BFGS-B', bounds=bounds)
                    #res = so.minimize(ADF_loss,params_in,args=args, method='BFGS')
                    #print (res)
                    params_out = res.x

                    #compute the difference between the simulated amplitude at one antenna and the reconstructed one 
                    
                    amplitude_diff = ADF_residuals(params_out, co.peak_amp_array[current_recons,:co.nants[current_recons]], co.antenna_coords_array[current_recons,:co.nants[current_recons]], Xmax, asym_coeff=0.01)
                    amplitude_recons = ADF_simulation(params_out,co.peak_amp_array[current_recons,:co.nants[current_recons]], co.antenna_coords_array[current_recons,:co.nants[current_recons]], Xmax, asym_coeff=0.01)
                    eta, omega = ADF_omega_eta(params_out,co.peak_amp_array[current_recons,:co.nants[current_recons]], co.antenna_coords_array[current_recons,:co.nants[current_recons]], Xmax, asym_coeff=0.01)
                    st.write_amplitude_residuals(st.outfile_res, co.coinc_index_array[current_recons, 0], co.nants[current_recons], co.peak_amp_array[current_recons,:co.nants[current_recons]], amplitude_diff, amplitude_recons, eta*180/np.pi, omega*180/np.pi, co.antenna_coords_array[current_recons,:co.nants[current_recons]])
                    # Compute errors with numerical estimates of Hessian matrix, inversion and sqrt of diagonal terms
                    # hess = nd.Hessian(ADF_loss)(params_out,*args)
                    # errors = np.sqrt(np.diag(np.linalg.inv(hess)))
                    errors = np.array([np.nan]*4)
                    print ("Best fit parameters = ",*np.rad2deg(params_out[:2]),*params_out[2:])
                    print ("Chi2 at best fit = ",ADF_loss(params_out,*args))
                    print ("Errors on parameters (from Hessian) = ",*np.rad2deg(errors[:2]),*errors[2:])
                    end_time = time.time()
                    adf_time = end_time - begining_time
                    st.write_adf(st.outfile,co.coinc_index_array[current_recons,0],co.nants[current_recons],params_out,errors,ADF_loss(params_out,*args), adf_time)
            except ZeroDivisionError as erreur:
                st.write_adf_false(st.outfile, co.coinc_index_array[current_recons,0])
                #st.write_adf(st.outfile, co.coinc_index_array[current_recons,0], -1, [-1, -1, -1, -1], [-1, -1, -1, -1], -1, -1)
                print("Une division par zéro a été détectée :", erreur)

        return

if __name__ == '__main__':

    main()

