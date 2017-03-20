import matplotlib.pyplot as py
import numpy as np
from scipy.integrate import odeint
#  ----------------------------------------------------------
# CONSTANTS
k = 1.38*(10**(-23))
hbar = (6.62607004*(10**(-34))/(2*np.pi)
c = 3*(10**8)
Stef_bltz = 5.67*(10**(-8))
a = (4.0*Stef_bltz)/c
mp = 1.67*(10**(-27))
me = 9.11*(10**(-31))
G = 6.67*(10**(-11))
M_s = 1.989*(10**30)
AU = 1.49*(10**11)

P_const = ((3*(np.pi**2))**(2.0/3.0))/5.0 # Constant Prefactors in P eq'n
P_term1 = P_const*((hbar**2)/me)*((p/mp)**(5.0/3.0))
P_term2 = ((p*k*T)/(mu*mp)) 
P_term3 = (1.0/3.0)*a*(T**4)
P_total = P_term1 + P_term2 + P_term3

mu = 1.0/(2*X +0.75*Y +0.5*Z)

dPdp_const = ((3*(np.pi**2))**(2.0/3.0))/3.0
dPdp_term1 = dPdp_const * ((hbar**2)/(me*mp)) * ((p/mp)**(2.0/3.0))
dPdp_term2 = (k*T)/(mu*mp)
dPdp_total = dPdpterm1 + dPdpterm2

dPdT_term1 = (p*k)/(mu*mp)
dPdT_term2 = (4.0/3.0)*a*(T**3)
dPdT_total = dPdT_term1 + dPdT_term2

Eps_PP = 1.07*(10**(-7))*p5*(X**2)*(T6**4)
Eps_CNO = 8.24*(10**(-26))*p5*X*X_CNO*(T6**4)
Eps = Eps_PP + Eps_CNO

kap_es = 0.02*(1 + X) 
kap_ff = 1.0 * (10**24)*(Z + 0.0001)*(p3**(0.7))*(T**(-3.5))
kap_H = (2.5 * (10**(-32)))*(Z/0.02)*(p3**(0.5))*(T**9)

kap_total = 1.0/((1.0/kap_H)+(1.0/(max(kap_es,kap_ff))))

p3 = p/(10**3)
p5 = p/(10**5)
T6 = T/(10**6)
X_CNO = 0.03*X

X = 0.73
Y = 0.25
Z = 0.02

def odefxn(y,r):
	p, T, M, L, tau = y
	yprime = 

dpdr = -(((G*M*p)/(r**2)) + )

dPdT = ((p*k)/(mu*mp)) + (1.0/12.0)*a*(T**3)


#  ----------------------------------------------------------