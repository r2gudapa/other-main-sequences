# coding: utf-8

# In[15]:
import constants as ct
import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt

np.seterr(divide='ignore')

# Boundary Conditions - fix at r_0 later
p_c = 1.5e5 #kg m^-3
T_c = 15e6 #K
r_0 = 0.001 #m

p = p_c
T = T_c

p_3 = (p/(10*3))

kappa_es = 0.02*(1+ct.X)
kappa_ff = 1.0*(10**24)*(ct.Z + 0.0001)*((p_3)**0.7)*(T**(-3.5))
kappa_H = 2.5*(10**(-32))*(ct.Z/0.02)*(p_3**0.5)*(T**9)

kap = (1/kappa_H + 1/(max(kappa_es,kappa_ff)))
                    

eppcoeff = (1.07*(10**(-7)))
ecnocoeff = (8.24*(10**(-26)))

# Boundary Conditions - fix at r_0 later
p_c = 1.5e5
T_c = 15.0e06
r_0 = 0.001
p_5 = (p/(10**5.0))
T_6 = (T/(10**6.0))
X_cno = 0.03*ct.X
e_pp = eppcoeff*p_5*(ct.X**2)*(T_6**4.0)
e_cno = ecnocoeff*p_5*ct.X*X_cno*(T_6**(19.9))

eps = e_pp + e_cno

M = ((4.0*ct.pi)/3.0)*((r_0)**3)*p_c
L = ((4.0*ct.pi)/3.0)*((r_0)**3)*p_c*eps

mu = (2.0*ct.X + 0.75*ct.Y + 0.5*ct.Z)**(-1)

P_1 = (((4*ct.pi**2)**(2.0/3.0))/5.0)*((ct.hbar**2)/ct.m_e)*((p/ct.m_p)**(5.0/3))
P_2 = p*(ct.k*T)/(mu*ct.m_p)
P_3 = (1.0/3)*ct.a*T**4.0

P = P_1 + P_2 + P_3

dPdp_1 = (((3*ct.pi**2)**(2.0/3.0))/3.0)*(ct.hbar**2/(ct.m_e*ct.m_p))*((p/ct.m_p)**(2.0/3.0))
dPdp_2 = (ct.k*T)/(mu*ct.m_p)
dPdp = dPdp_1 + dPdp_2

dPdT = p*(ct.k/(mu*ct.m_p))+(4.0/3.0)*ct.a*(T**3)




def vectorfield(w,r):
    
    p, T, M, L, tau = w
    
    P_1 = (((4*ct.pi**2)**(2.0/3.0))/5.0)*((ct.hbar**2)/ct.m_e)*((p/ct.m_p)**(5.0/3))
    P_2 = p*(ct.k*T)/(mu*ct.m_p)
    P_3 = (1.0/3)*ct.a*T**4.0
    P = P_1 + P_2 + P_3

    dPdp_1 = (((3*ct.pi**2)**(2.0/3.0))/3.0)*(ct.hbar**2/(ct.m_e*ct.m_p))*((p/ct.m_p)**(2.0/3.0))
    dPdp_2 = (ct.k*T)/(mu*ct.m_p)
    dPdp = dPdp_1 + dPdp_2

    dPdT = p*(ct.k/(mu*ct.m_p))+(4.0/3.0)*ct.a*(T**3)
    
    dTdr=-min(((3*kap*p*L)/(16*ct.pi*ct.a*ct.c*T**3*r**2)), (1-1/ct.gamma)*(T/P)*(ct.G*M*p/r**2))
    dpdr=(-(ct.G*M*p/(r**2)+dPdT*dTdr)/dPdp)
    dMdr=4*ct.pi*r**2*p
    dLdr=4*ct.pi*r**2*p*eps
    dtaudr=kap*p
    # f = [dpdr,dTdr, dMdr, dLdr, dtaudr]
    
    
    f = [dpdr,
        dTdr,
        dMdr,
        dLdr,
        dtaudr]
    return f

r = np.linspace(0.0000001, ct.R_s, 10)  
y0 = [p_c,T_c, 0.0, 0.0, 1.0e6]


solver=ode(vectorfield)
solver.set_integrator('dopri5')
sol = solver.set_initial_value(y0,r[0])
sol=np.empty((len(r),5))
sol[0] = y0
   
d=1
while solver.successful() and solver.r<ct.R_s:
    solver.integrate(r[d])
    sol[d]=solver.f
    k += 1

axes = plt.gca()
axes.set_xlim(0,3.5e4)
plt.plot(r, sol[:, 0], label='rho')
#plt.plot(r, sol[:, 1], label='temp')
#plt.plot(r, sol[:, 2], label='Mass')
plt.plot(r, sol[:, 3], label='Lum')
#plt.plot(r, sol[:, 4], label='tau')
plt.grid()
plt.legend()
  


plt.figure(1)
plt.title("Graph 1", fontsize=25)
plt.xlabel('r(m)', fontsize=20)
plt.ylabel('$P_s$', fontsize=20)
plt.savefig('Ps vs r', dpi=1000)
plt.show()