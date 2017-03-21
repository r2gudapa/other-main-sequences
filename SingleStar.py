import constants as ct
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

Kes = 0.02*(1.0 + ct.X) #m^2/kg/m^3 #electron scattering opacity
X_cno = 0.03*ct.X #fraction of H going under CNO
mu = (2.0*ct.X + 0.75*ct.Y + 0.5*ct.Z)**(-1) #mean molecular weight for 100% ionization

#boundary conditions
p_c = 1.5e6 #kg/m^3
T_c = 1.6e7 #K
r0 = 0.000001 #m

class SingleStar:
	
	def __init__(self, dr, rho_central, T_central):
	#create a function for rk4
	
		self.dr = dr
		
	############___Pressure Functions___############
	
	#Degeneracy pressure
	def P_deg(self, density):
		return ((3*np.pi**2)**(2.0/3.0)*ct.hbar**2*(density/ct.m_p)**(5.0/3.0))/(5.0*ct.m_e)
	
	#Ideal gas pressure
	def P_ig(self,density,temp):
		return (density*ct.k*temp)/(mu*ct.m_p)
		
	#Radiative pressure
	def P_rad(self,temp):
		return (1.0/3.0)*ct.a*temp**4
	
	#total pressure
	def P(self):
		return self.P_deg(density) + self.P_ig(density,temp) + self.P_rad(temp)
		
	#dP/dp
	def dPdp(self, density, temp):
	
		dPdp_deg = ((3*np.pi**2)**(2.0/3.0)*ct.hbar**2*(density/ct.m_p)**(2.0/3.0))/(3.0*ct.m_e*ct.m_p)
		dPdp_ig = (ct.k*temp)/(mu*ct.m_p)
		
		return dPdp_deg + dPdp_ig
		
	#dP/dT
	def dPdT(self,density,temp):
	
		dPdT_ig = (density*ct.k)/(mu*ct.m_p)
		dPdT_rad = (4.0/3.0)*ct.a*temp**3
		
		return dPdT_ig + dPdT_rad
	
	############___System of Equations___############
	
	#density differential eqn
	def dpdr(self, mass, density, radius, temp, luminosity):
		return -((ct.G*mass*density)/(radius**2) + self.dPdT(density,temp)*self.dTdr(mass,density,temp,radius,luminosity))/self.dPdp(density,temp)
	
	#temp differential eqn
	def dTdr(self,mass,density,temp,radius,luminosity):
		return -min(self.dTdr_rad(density,luminosity,temp,radius),self.dTdr_conv(temp,mass,density,radius))
		
	def dTdr_rad(self,density,luminosity,temp,radius)
		return (3.0*self.Kappa(density,temp)*density*luminosity)/(16.0*np.pi*ct.a*ct.c*temp**3*radius**2)
		
	def dTdr_conv(self,temp,mass,density,radius)
		return (1.0 - (1.0/ct.gamma))*(temp*ct.G*mass*density)/(self.P*radius**2)
		
	#mass differential eqn
	def dMdr(self,radius,density):
		return 4.0*np.pi*radius**2*density
		
	#luminosity differential eqn
	def dLdr(self,radius,density)
		return 4.0*np.pi*radius**2*density*self.Epsilon(density,temp)
		
	#optical depth differential eqn
	def dtaudr(self,density):
		return self.Kappa(density,temp)*density
	
	############___Opacity Equations___############	
	
	#free-free scattering opacity
	def Kff(self,density,temp):
		return (1.0e24)*(ct.Z+0.0001)*(density/1e3)**(0.7)*temp**(-3.5)
	
	#hydrogen ion opacity 	
	def KH(self,density,temp):
		return (2.5e-32)*(ct.Z/0.02)*(density/1e3)**(0.5)*temp**9
	
	#overall opacity
	def Kappa(self,density,temp):
		if temp > 1.0e4:
			Kappa = ((1.0/self.KH(density,temp))+(1.0/max(Kes,self.Kff(density,temp))))
			
		else:
			Kappa = ((1.0/self.KH(density,temp))+(1.0/min(Kes,self.Kff(density,temp))))
			
		return Kappa
	
	############___Energy Generation Equations___############
	
	#PP-chain energy generation rate
	def epsilonPP(self,density,temp):
		return (1.07e-7)*(density/1.0e5)*ct.X**2*(temp/1.0e6)**4
	
	#CNO cycle energy generation rate
	def epsilonCNO(self,density,temp):
		return (8.24e-26)*(density/1.0e5)*ct.X*X_cno*(temp/1.0e6)**(19.9)
		
	#overall energy generation rate
	def Epsilon(self,density,temp):
		return self.epsilonPP(density,temp) + self.epsilonCNO(density,temp)
		
	
	
	
	
		
		
		
		
		