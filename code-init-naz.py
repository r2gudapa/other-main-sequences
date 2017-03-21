# coding: utf-8

# In[15]:
import constants as ct
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

np.seterr(divide='ignore')

# Boundary Conditions - fix at r_0 later
p_c = 1.5e5 #kg m^-3
T_c = 15e6 #K
r_0 = 0.001 #m

p = p_c
T = T_c
dr=0.01

p_3 = (p/(10*3))
p_5 = (p/(10**5.0))
T_6 = (T/(10**6.0))
X_cno = 0.03*ct.X
                    
eppcoeff = (1.07*(10**(-7)))
ecnocoeff = (8.24*(10**(-26)))

e_pp = eppcoeff*p_5*(ct.X**2)*(T_6**4.0)
e_cno = ecnocoeff*p_5*ct.X*X_cno*(T_6**(19.9))


eps = e_pp + e_cno
kappa_es = 0.02*(1+ct.X)
kappa_ff = 1.0*(10**24)*(ct.Z + 0.0001)*((p_3)**0.7)*(T**(-3.5))
kappa_H = 2.5*(10**(-32))*(ct.Z/0.02)*(p_3**0.5)*(T**9)

kap = (1/kappa_H + 1/(max(kappa_es,kappa_ff)))

#Initial M and L values (rest should be iterated within vectorfield -N)
M = ((4.0*ct.pi)/3.0)*((r_0)**3)*p_c       
L = ((4.0*ct.pi)/3.0)*((r_0)**3)*p_c*eps


mu = (2.0*ct.X + 0.75*ct.Y + 0.5*ct.Z)**(-1)

#Initial P and dPdT and dPdP values (shouldn't it be in vectorfield? Doesnt seem to be changing -N)
P_1 = (((4*ct.pi**2)**(2.0/3.0))/5.0)*((ct.hbar**2)/ct.m_e)*((p/ct.m_p)**(5.0/3.0))
P_2 = p*(ct.k*T)/(mu*ct.m_p)
P_3 = (1.0/3)*ct.a*T**4.0

P = P_1 + P_2 + P_3

dPdp_1 = (((3.0*ct.pi**2)**(2.0/3.0))/3.0)*(ct.hbar**2/(ct.m_e*ct.m_p))*((p/ct.m_p)**(2.0/3.0))
dPdp_2 = (ct.k*T)/(mu*ct.m_p)
dPdp = dPdp_1 + dPdp_2

dPdT = p*(ct.k/(mu*ct.m_p))+(4.0/3.0)*ct.a*(T**3)

#def updateval(val, rk4_out,rad_n):
    
    


def vectorfield(w,r):
    
    p, T, M, L, tau = w
    '''
    print "-----"
    print "kap=",kap
    print "p=",p
    print "T=", T
    print "M=", M
    print "L=", L
    print "r=", r
    '''
#    ct.G, kap, ct.pi, ct.a, ct.c, ct.gamma, eps = q
    dTdr=-min(((3.0*kap*p*L)/(16.0*ct.pi*ct.a*ct.c*T**3.0*r**2.0)), (1.0-1.0/ct.gamma)*(T/P)*(ct.G*M*p/r**2.0))

    '''
    dTdr_1=((3.0*kap*p*L)/(16.0*ct.pi*ct.a*ct.c*(T**3.0)*(r**2.0)))
    dTdr_2=(1.0-1.0/ct.gamma)*(T/P)*(ct.G*M*p/r**2.0)
    print "dTdr_1=", dTdr_1
    print "dTdr_2=", dTdr_2
    '''
    
    # f = [dpdr, dTdr, dMdr, dLdr, dtaudr]
    f = [(-(ct.G*M*p/(r**2)+dPdT*dTdr)/dPdp),
        dTdr,
        4.0*ct.pi*r**2.0*p,
        4.0*ct.pi*r**2.0*p*eps,
        kap*p]
    return f

  
r = np.linspace(0.0001, ct.R_s, 10000)  #Iterate Radius
y0 = [p_c,T_c, M, L, 1.0e6] #Create values to send to vectorfield (p, T, M, L, tau)
sol=odeint(vectorfield, y0, r) #Calling odeint (We should look into changing it to rk4-N)


#Plot values have been normalised similarly to the graphs provided in the project description file.
#Can anyone conver the graph code to show all of them simultaneously and seperately? Not sure how to do so -N
#Right now have to uncomment the graph that you want to see
plt.figure(1)
axes = plt.gca()
#axes.set_xlim(0,3.5e4)
plt.plot(r, sol[:, 0]/p_c, label='rho')
#plt.plot(r, sol[:, 1]/T_c, label='temp')
#plt.plot(r, sol[:, 2]/(ct.M_s), label='Mass')
#plt.plot(r, sol[:, 3]/(ct.L_s), label='Lum')
#plt.plot(r, sol[:, 4], label='tau')

#Finding Surface Values to compare to document provided
print "Surface Density=", sol[len(sol[:,0])-1,0], "kg/m$\^$3"
print "Surface Temp=", sol[len(sol[:,1])-1,1], "K"
print "Surface Mass=", sol[len(sol[:,2])-1,2]/ct.M_s, "M_s"
print "Surface Luminosity=", sol[len(sol[:,3])-1,3]/ct.L_s, "L_s"
print "Surface Optical Depth=", sol[len(sol[:,4])-1,4], "??" #Should be 2/3? -N

plt.grid()
plt.legend()
plt.title("Graph 1", fontsize=25)
plt.xlabel('r(m)', fontsize=20)
plt.ylabel('$y$', fontsize=20)
plt.savefig('Test', dpi=1000)
plt.show()
