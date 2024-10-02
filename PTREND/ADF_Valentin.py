import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import sys



kc = 2.997924580e8
kepsilon_0 = 8.85418782e-12
kvolt_ev = 1.6e-19
kt_sampling = 2.e-9
kn = 1.0022

GroundAltitude = 1086
bFieldDecl = 0.
bFieldIncl = np.pi/2. + 1.0609856522873529

def GetEffectiveRefractionIndex(x0,y0,z0,ns=325,kr=-0.128,zant=0,xant=0,yant=0,stepsize = 20000): #NOTE THAT THE ORDER OF ANTENNA POSITIONS IS ZANT,XANT,YANT (historic compatibility reasons)
        rearth=6370949.0
        R02=x0*x0+y0*y0  #notar que se usa R02, se puede ahorrar el producto y la raiz cuadrada
        h0=(np.sqrt((z0+rearth)*(z0+rearth) + R02 ) - rearth)/1.E3 #altitude of emission, in km

        rh0 = ns*np.exp(kr*h0) #refractivity at emission
        n_h0=1.E0+1.E-6*rh0 #n at emission
#        print("n_h0",n_h0,ns,kr,x0,y0,z0,h0,rh0)

        hd=(zant)/1.E3 #detector altitude

#       Vector from detector to average point on track. Making the integral in this way better guaranties the continuity
#       since the choping of the path will be always the same as you go farther away. If you start at your starting point, for a given geometry,
#       the choping points change with each starting position.

        ux = x0-xant
        uy = y0-yant         #the antenna position, considered to be at the core
        uz = z0-zant

        Rd=np.sqrt(ux*ux + uy*uy)
        kx=ux/Rd
        ky=uy/Rd #!k is a vector from the antenna to the track, that when multiplied by Rd will end in the track and sumed to antenna position will be equal to the track positon
        kz=uz/Rd

#       integral starts at ground
        nint=0
        sum=0.E0

        currpx=0+xant
        currpy=0+yant    #!current point (1st antenna position)
        currpz=zant
        currh=hd


        while(Rd > stepsize): #if distance projected on the xy plane is more than 10km
          nint=nint+1
          nextpx=currpx+kx*stepsize
          nextpy=currpy+ky*stepsize           #this is the "next" point
          nextpz=currpz+kz*stepsize

          nextR2=nextpx*nextpx + nextpy*nextpy #!se usa el cuadrado, se puede ahorrar la raiz cuadrada
          nexth=(np.sqrt((nextpz+rearth)*(nextpz+rearth) + nextR2) - rearth)/1.E3

          if(np.absolute(nexth-currh) > 1.E-10  ):   #check that we are not going at constant height, if so, the refraction index is constant
              sum=sum+(np.exp(kr*nexth)-np.exp(kr*currh))/(kr*(nexth-currh))
          else:
              sum=sum+np.exp(kr*currh)

          currpx=nextpx
          currpy=nextpy
          currpz=nextpz  #Set new "current" point
          currh=nexth

          Rd=Rd-stepsize #reduce the remaining lenght
        #enddo

        #when we arrive here, we know that we are left with the last part of the integral, the one closer to the track (and maybe the only one)

        nexth=h0

        if(np.absolute(nexth-currh) > 1.E-10 ): #check that we are not going at constant height, if so, the refraction index is constant
          sum=sum+(np.exp(kr*nexth)-np.exp(kr*currh))/(kr*(nexth-currh))
        else:
          sum=sum+np.exp(kr*currh)

        nint=nint+1
        avn=ns*sum/nint
        n_eff=1.E0+1.E-6*avn #average (effective) n
        return n_eff

def GetRefractionIndexAtXmax(x0, y0, z0, ns,kr):
    rearth=6370949.0
    R02=x0*x0+y0*y0  #notar que se usa R02, se puede ahorrar el producto y la raiz cuadrada
    h0=(np.sqrt((z0+rearth)*(z0+rearth) + R02 ) - rearth)/1.E3 #altitude of emission, in km

    rh0 = ns*np.exp(kr*h0) #refractivity at emission
    n_h0=1.E0+1.E-6*rh0 #n at
    return n_h0

def get_in_shower_plane(pos_, k_, core_decay_, inclination_, declination_):

    _pos = (pos_ - core_decay_).T
    _B = np.array([np.cos(declination_)*np.sin(inclination_), np.sin(declination_)*np.sin(inclination_),np.cos(inclination_)])
    _kxB = np.cross(k_,_B)
    _kxB /= np.linalg.norm(_kxB)
    _kxkxB = np.cross(k_,_kxB)
    _kxkxB /= np.linalg.norm(_kxkxB)
    #print("k_", k_, "_kxB = ", _kxB, "_kxkxB = ", _kxkxB)

    return np.array([np.dot(_kxB, _pos), np.dot(_kxkxB, _pos), np.dot(k_, _pos)])

def master_equation(w_, X_, Delta_, alpha_, n0_, n1_):
    _eq = X_**2 * np.sin(alpha_)**2 * (n0_**2 - n1_**2) + 2.*Delta_ * X_ * np.sin(alpha_) * (n0_ - n1_**2*np.cos(w_))*np.sin(alpha_ - w_) + Delta_**2 * (1. - n1_**2) * np.sin(alpha_ - w_)**2
    return _eq

def compute_observer_position(w_, k_, u_, x_Xmax_, y_Xmax_, z_Xmax_, GroundAltitude_):
    rot_vect = np.cross(u_, k_)
    rot_vect /= np.linalg.norm(rot_vect)
    vect_dir = np.array([np.dot(rotation_matrix(rot_vect,w_)[:,0], k_), np.dot(rotation_matrix(rot_vect,w_)[:,1], k_), np.dot(rotation_matrix(rot_vect,w_)[:,2], k_)])
    _t = (GroundAltitude_ - z_Xmax_) / vect_dir[2]
    _x = x_Xmax_+vect_dir[0]*_t
    _y = y_Xmax_+vect_dir[1]*_t
    _z = z_Xmax_+vect_dir[2]*_t
    return _x, _y, _z

def solve_dichotomi(w_start, max_it_, k_, eta_, X_, Delta_, x_Xmax_, y_Xmax_, z_Xmax_, GroundAltitude_):

    #compute the emission point before Xmax
    _x_Before = x_Xmax_ - k_[0] * Delta_
    _y_Before = y_Xmax_ - k_[1] * Delta_
    _z_Before = z_Xmax_ - k_[2] * Delta_
    #compute the direction vector
    k_plan = np.array([k_[0], k_[1], 0]) /np.sqrt(k_[0]**2 + k_[1]**2)
    rot_vect = rotation_matrix(np.array([0, 0, 1]), -eta_)
    _u = np.array([np.dot(rot_vect[:,0], k_plan), np.dot(rot_vect[:,1], k_plan), np.dot(rot_vect[:,2], k_plan)])
    #Compute the alpha angle
    _alpha =  np.arccos(np.dot(k_, _u))

    _w_test = w_start
    _w_step = 0.0001
    _res = -1.e3

    _it = 0
    while(_res < 0):

        _x, _y, _z = compute_observer_position(_w_test, k_, _u, x_Xmax_, y_Xmax_, z_Xmax_, GroundAltitude_)
        _n0 = GetEffectiveRefractionIndex(x_Xmax_, y_Xmax_, z_Xmax_, 325.0, -0.1218, GroundAltitude_, xant=_x,yant=_y,stepsize = 200)
        _n1 = GetEffectiveRefractionIndex(_x_Before, _y_Before, _z_Before, 325.0, -0.1218, GroundAltitude_, xant=_x,yant=_y,stepsize = 200)
        _res = master_equation(_w_test, X_, Delta_, _alpha, _n0, _n1)
        _w_test += _w_step
        _it+=1
        if (_it > max_it_):
            print("Max iteration reached: stopping dichotmi search")
            _w_test = 0.
            continue

    return (_w_test-_w_step)

def rotation_matrix(u, angle):
    #compute a rotation of direction: u and angle: angle
    rot_M = np.zeros((3, 3))
    rot_M[0,:] = [np.cos(angle) + u[0]**2*(1. - np.cos(angle)), u[0]*u[1]*(1. - np.cos(angle)) - u[2]*np.sin(angle), u[0]*u[2]*(1. - np.cos(angle)) + u[1]*np.sin(angle)]
    rot_M[1,:] = [ u[1]*u[0]*(1. - np.cos(angle)) + u[2]*np.sin(angle), np.cos(angle) + u[1]**2*(1. - np.cos(angle)), u[1]*u[2]*(1. - np.cos(angle)) - u[0]*np.sin(angle)]
    rot_M[2,:] = [u[2]*u[0]*(1. - np.cos(angle)) - u[1]*np.sin(angle),  u[2]*u[1]*(1. - np.cos(angle)) + u[0]*np.sin(angle), np.cos(angle) + u[2]**2*(1. - np.cos(angle))]
    return rot_M

def f_cerenkov(w_, l_, alpha_, theta_, wc_, dw_):
    _dw = dw_
    #_dw = np.cos(theta_)/ (dX[2]/l_) * dw_
    #_dw = (np.cos(np.pi - theta_)/np.cos(np.pi - alpha_))*dw_
    _ratio = np.tan(w_)/np.tan(wc_)
    return 1. / ( 1.+ 4.*((_ratio)**2 - 1)**2/_dw**2)


def f_asymetry(eta_, sin_chi_):
    return 1. + sin_chi_*0.005 *np.cos(eta_)

def adf_model(w_, l_, eta_, alpha_, sin_chi_, theta_, wc_, dw_, a_):
    return a_/l_*f_cerenkov(w_, l_, alpha_, theta_, wc_, dw_) * f_asymetry(eta_, sin_chi_)

def ADF(xant_, yant_, zant_, xxmax_, yxmax_, zxmax_, azim_, zen_, bFieldIncl_, bFieldDecl_, grd_alt_, xmaxdist_, cerenkov_width_, adf_amp_):
    #B en radians
    #theta et phi en ° en GRAND conventions
    _k = np.array([np.cos(azim_*np.pi/180.)*np.sin(zen_*np.pi/180),np.sin(azim_*np.pi/180.)*np.sin(zen_*np.pi/180), np.cos(zen_*np.pi/180)]).T

    xsp, ysp, zsp = get_in_shower_plane(np.array([xant_, yant_, zant_]).T, _k, np.array([xxmax_, yxmax_, zxmax_]), bFieldIncl_, bFieldDecl_)

    ##In array frame
    _obs = np.array([xant_ - xxmax_, yant_ - yxmax_, zant_ - zxmax_])
    _l = np.sqrt((xant_ - xxmax_)**2 + (yant_ - yxmax_)**2 + (zant_ - zxmax_)**2)
    _long = np.dot(_k, _obs)
    cerenkov_width_ = np.cos(np.deg2rad(zen_))/ (_obs[2]/_l) * cerenkov_width_
    _uant = _obs / _l
    _w = np.arccos(np.dot(_k, _uant))

    _eta = np.arctan2(ysp, xsp)
    _max_eta = max(_eta)
    _min_eta = min(_eta)

    _u1 = _uant.T - _k
    _u1[:,2] = 0
    _u = np.array([_u1[i,:]/np.linalg.norm(_u1[i,:]) for i in range(len(xant_))])
    _k_pa = np.array([_k[0], _k[1], 0])
    _k_pa /= np.sqrt(_k_pa[0]**2+_k_pa[1]**2)
    _xi = np.pi - np.arccos(np.dot(_u, -_k_pa))

    _obs_bis = np.array([_obs[0,:], _obs[1,:]]) / _l
    _xi_bis = np.pi - np.arccos((-_k[0]*(_obs_bis[0,:] - _k[0]) - _k[1]*(_obs_bis[1,:] - _k[1]))/(np.sqrt((_obs_bis[0,:] - _k[0])**2 + (_obs_bis[1,:] - _k[1])**2)*np.sqrt(_k[0]**2 + _k[1]**2)));

    _B = np.array([np.sin(bFieldIncl_), 0,np.cos(bFieldIncl_)])
    _sin_chi = 1. - np.dot(_k, _B)**2

    _alpha = np.arccos(_uant[2,:])
    _n = np.array([GetEffectiveRefractionIndex(xxmax_, yxmax_, zxmax_,325.0,-0.1218,grd_alt_,xant=xant_[i],yant=yant_[i],stepsize = 20000) for i in range(0, len(xant_))])
    _n_xmax = GetRefractionIndexAtXmax(xxmax_,yxmax_,zxmax_,325.0,-0.1218)

    ##Compute Cerenkov table
    _eta_list = np.linspace(0, 180, 19)*np.pi/180.
    _cerenkov_table = np.array([solve_dichotomi(0., 1.e4, _k, e, xmaxdist_, 2.e3, xxmax_, yxmax_, zxmax_, grd_alt_) for e in _eta_list])

    ##Get Cerenkov angles
    _cerenkov = np.zeros(len(_w))
    for i in range(len(_xi)):
        sel = np.where(np.abs(_xi[i] - _eta_list) == min(np.abs(_xi[i] - _eta_list)))[0]
        _cerenkov[i] = _cerenkov_table[sel]
        #_cerenkov[i] = np.arccos(1/_n_xmax)
    print('Cherenkov',_cerenkov)

    return np.array([_w*180/np.pi, _eta*180/np.pi, adf_model(_w, _l, _eta, _alpha, _sin_chi, zen_, _cerenkov, cerenkov_width_, adf_amp_)])
