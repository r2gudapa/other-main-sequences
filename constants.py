import numpy as np

# ALL CONSTANTS BELOW IN SI UNITS

# Proton Mass
m_p = 1.6726219e-27 

# Electron Mass
m_e = 9.109e-31 

# Mass of the Sun
M_s = 1.989e30 

# Radius of the Sun
R_s = 6.963e8 

# Luminosity of the Sun
L_s = 3.828e26 

# Big G = gravitational constant
G = 6.674e-11 

# speed of light
c = 2.998e8   

# Boltzmann Constant
k = 1.381e-23

# Stefan-Boltzmann Constant
stfb = 5.670e-8 

# Radiation Energy Density Constant
a = 4.0 * stfb / c

# Just pi
pi = np.pi

# Adiabatic Index for Ideal Gas
gamma = 5.0/3.0 

# Planck's Constant
h = 6.626e-34 

# Reduced Planck's Constant
hbar = h / (2 * np.pi)

# HI -> HII ionization energy (IN eV)
chi = 13.6

############################################

# VALUES OF MEAN MOLECULAR WEIGHT

# fully ionized pure-H gas
mu_HII = 1.0/2.0

# fully ionized pure-He gas
mu_HeIII = 4.0/3.0

# Sun-like star
mu_S = 0.612

# Red Giant 
mu_RG = 1.3 # (please check)

############################################

#COMPOSITION VALUES X Y Z

# X - Hydrogen
X= 0.73

# Y - Helium
Y= 0.25

# Z - Metals/Other
Z= 0.02