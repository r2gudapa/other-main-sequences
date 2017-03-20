
# coding: utf-8

# In[15]:

import constants as ct
import numpy as np
from scipy.integrate import odeint

eppcoeff = (1.07*(10**(-7)))
ecnocoeff = (8.24*(10**(-26)))

P_1 = (((4*ct.pi**2)**(2.0/3.0))/5.0)*((ct.hbar**2)/ct.m_e)*((p/ct.m_p)**(5.0/3))
P_2 = p*(k*T)/(mu*ct.m_p)
P_3 = (1.0/3)*a*T**4.0

P = P_1 + P_2 + P_3

mu = (2.0*X + 0.75*Y + 0.5*Z)**(-1)

dPdp_1 = (((3*ct.pi**2)**(2.0/3))/3)*(ct.hbar**2/(ct.m_e*ct.m_p))(*(p/ct.m_p)**(2.0/3))
dPdp_2 = (kT)/(mu*ct.m_p)
dPdp = dPdp_1 + dPdp_2

dPdT = p*(k/(mu*ct.m_p))+(4.0/3)*a*(T**3)

p_5 = (p/(10**5.0))
T_6 = (T/(10**6.0))
X_cno = 0.03*X

e_pp = eppcoeff*p_5*(X**2)*(T_6**4.0)
e_cno = ecnocoeff*p_5*X*X_cno*(T_6**(19.9))

eps = e_pp + e_cno

def vectorfield(p,T,M,L,tau):
    
    p, P, T, M, L, tau, = w
    G, kappa, ct.pi, a, c, gamma, eps = p
    
    # f = [dPdr, dpdr, dTdr_rad, dTdr_conv, dMdr, dLdr, dtaudr]
    f = [-(G*M*p/(r**2))
        (-(G*M*p/(r**2)+dPdT*dTdr)/dPdp),
        (3*kappa*p*L)/(16*ct.pi*a*c*T**3*r**2),
        (1-1/gamma)*(T/P)*(G*M*p/r**2),
        4*ct.pi*r**2*p,
        4*ct.pi*r**2*p*eps,
        kappa*p]
    return f

# Boundary Conditions - fix at r_0 later
M = 0
L = 0
p = p_c
T = T_c

p_3 = (p/(10*3))

kappa_es = 0.02*(1+X)
kappa_ff = 1.0*(10**24)*(Z + 0.0001)*((p_3)**0.7)*(T**(-3.5))
kappa_H = 2.5*(10**(-32))(Z/0.02)*(p_3**0.5)*(T**9)

kap = (1/kappa_H + 1/(max(kappa_es,kappa_ff)))
                    


# In[ ]:



