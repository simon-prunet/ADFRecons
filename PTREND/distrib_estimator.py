'''

'''
from wavefronts import *
from recons import *
import matplotlib.pyplot as plt


path_pos = "../Chiche/coord_antennas.txt"
path_event = "../Chiche/Rec_coinctable.txt"
path_guess_SWF = "../Chiche/Rec_plane_wave_recons_py.txt"


def read_event_add_noise(path_event, path_pos, nb_event_noised, sigma_t, sigma_val=0):
    
    o_an = antenna_set(path_pos)
    co = coincidence_set(path_event, o_an)
    an_pos = co.antenna_coords_array[0,:]
    
    no_noise_time = co.peak_time_array[0,:]
    nb_du = no_noise_time.shape[0]
    print(no_noise_time.shape)
    # convert sigma time to sigma distance
    noise = np.random.normal(0, c_light * sigma_t, nb_du * nb_event_noised)
    noise_time = no_noise_time + noise.reshape((nb_event_noised, nb_du))
    # add biais
    # noise_time  + = c_light * 10e-9             
    
    return an_pos, no_noise_time, noise_time


def distrib_SWF(path_guess, pos_du, peak_time_noised):
    '''
    
    :param path_guess:
    :param pos_du:
    :param peak_time_noised: array[nb_event_noised, nb_du]
    '''
    with open(path_guess) as fid_angles:
        l = fid_angles.readline().strip().split()
        theta_in, phi_in = np.float64(l[2]), np.float64(l[4])
    # Guess parameters
    bounds = [[np.deg2rad(theta_in - 1), np.deg2rad(theta_in + 1)],
             [np.deg2rad(phi_in - 1), np.deg2rad(phi_in + 1)],
             [-15.6e3 - 12.3e3 / np.cos(np.deg2rad(theta_in)), -6.1e3 - 15.4e3 / np.cos(np.deg2rad(theta_in))],
            [6.1e3 + 15.4e3 / np.cos(np.deg2rad(theta_in)), 0]]
    params_in = np.array(bounds).mean(axis=1)
    print ("params_in = ", params_in)
    nb_event = peak_time_noised.shape[0]
    # array best fit parameter
    a_sol = np.empty((nb_event, 3), dtype=np.float64)
    
    for idx_event in range(nb_event):
        # print(f'======= Process event {idx_event+1}')
        args = (pos_du, peak_time_noised[idx_event])
        res = so.minimize(SWF_loss, params_in.copy(), jac=SWF_grad, args=args, method="BFGS")
        a_sol[idx_event,:2 ] = np.rad2deg(res.x[:2])
        a_sol[idx_event, 2 ] = res.x[2]
        print (f"#{idx_event:04}# Best fit param :  {a_sol[idx_event,:2 ]}  {a_sol[idx_event, 2 ]}, Chi2= {SWF_loss(res.x, *args)}")
    return a_sol


def plot_dist_angle(angle1, angle2, title=""):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(title)
    ax1.set_title("azimth ? ")
    ax1.hist(angle1)
    ax1.set_xlabel(f'degree/%\nmean={angle1.mean()}\nstd={angle1.std()}')
    ax1.grid()
    ax2.set_title("elevation ?")
    ax2.hist(angle2)
    ax2.set_xlabel(f"degree/%\nmean={angle2.mean()}\nstd={angle2.std()}")
    ax2.grid()
    
def main_SWF(nb_tirage = 500, sigma_t = 0.1e-9):
    '''
    
    :param nb_tirage:
    :param sigma_t:  second
    '''
    an, no_noise_time, noise_time = read_event_add_noise(path_event, path_pos, nb_tirage, sigma_t)
    print(no_noise_time[:10])
    print(noise_time[0,:10])
    print(noise_time[1,:10])
    a_sol = distrib_SWF(path_guess_SWF, an, noise_time)
    title = f'Angles distribution ({nb_tirage} fit) for noised time (sigma={sigma_t/1e-9:4.2}ns), with SWF method' 
    plot_dist_angle(a_sol[:, 0], a_sol[:, 1], title)
    # true error
    true_angle1 = 117.02
    true_angle2 = 270.0
    title = f'True error angles distribution ({nb_tirage} fit) for noised time (sigma={sigma_t/1e-9:4.2}ns), with SWF method'
    plot_dist_angle(a_sol[:, 0]-true_angle1, a_sol[:, 1]-true_angle2, title)
    # relative error
    true_angle1 = 117.02
    true_angle2 = 270.0
    title = f'Relative error angles distribution ({nb_tirage} fit) for noised time (sigma={sigma_t/1e-9:4.2}ns), with SWF method'
    plot_dist_angle(100*(a_sol[:, 0]-true_angle1)/true_angle1, 
                    100*(a_sol[:, 1]-true_angle2)/true_angle2, title)
    
    
    
if __name__ == '__main__':
    main_SWF(1000, sigma_t =2e-9)
    plt.show()
