import numpy as np

# Physical constants
c_light = 2.997924580e8
R_earth = 6371007.0
ns = 325
kr = -0.1218
groundAltitude = 1264.0  # 1086.0
B_dec = 0.
B_inc = np.pi/2. + 1.0609856522873529
# Magnetic field direction (unit) vector
Bvec = np.array([np.sin(B_inc)*np.cos(B_dec), np.sin(B_inc)*np.sin(B_dec), np.cos(B_inc)])

D2R = np.pi / 180
R2D = 180 / np.pi