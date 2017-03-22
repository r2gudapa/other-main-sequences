import constants as ct
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

Kes = 0.02*(1.0 + ct.X) #m^2/kg/m^3 #electron scattering opacity
X_cno = 0.03*ct.X #fraction of H going under CNO
mu = (2.0*ct.X + 0.75*ct.Y + 0.5*ct.Z)**(-1) #mean molecular weight for 100% ionization

#boundary conditions
r0 = 0.000001 #m -- initial radius

class SingleStar:
	
	def __init__(self, dr, rho_central, T_central):
	
		#lists for things we need to plot, these will be filled as rk4 integrates the functions
		#to begin with these only include the initial conditions and as rk4 finds more values, it should append to these
		self.dr = dr #step size -- not constant since we are planning to use adaptive rk4, this will change
		self.radius = [r0]
		self.density = [rho_central] 
		self.temp = [T_central] 
		self.mass = [(4.0/3.0)*np.pi*self.radius[0]**3*self.density[0]] 
		self.lum = [(4.0/3.0)*np.pi*r0**3*self.density[0]*self.Epsilon(self.density[0],self.temp[0])] 
		
	#create a function for rk4
	def rk4(self,radius,density,temp,mass,lum,h):
		#Thank you Wikipedia!!
		#radius is the independent variable,
		#the rest are the dependent variables
		#h is step size

		h = self.dr
		
		#increment based on the slope at the beginning of the interval using yn
		k1 = self.dpdr(radius,density,temp,mass,lum)	
		l1 = self.dTdr(radius,density,temp,mass,lum)
		m1 = self.dMdr(radius,density)
		n1 = self.dLdr(radius,density,temp)
		o1 = self.dtaudr(density,temp)
		
		#increment based on the slope at the midpoint of the interval using yn + h/2*k1
		k2 = self.dpdr(radius + (h/2.0), density + (h/2.0)*k1, temp + (h/2.0)*l1, mass + (h/2.0)*m1, lum + (h/2.0)*n1)
		l2 = self.dTdr(radius + (h/2.0), density + (h/2.0)*k1, temp + (h/2.0)*l1, mass + (h/2.0)*m1, lum + (h/2.0)*n1)
		m2 = self.dMdr(radius + (h/2.0), density + (h/2.0)*k1)
		n2 = self.dLdr(radius + (h/2.0), density + (h/2.0)*k1, temp + (h/2.0)*l1)
		o2 = self.dtaudr(density + (h/2.0)*k1, temp + (h/2.0)*l1)
		
		#increment based on the slope at the midpoint of the interval using yn + h/2*k2
		k3 = self.dpdr(radius + (h/2.0), density + (h/2.0)*k2, temp + (h/2.0)*l2, mass + (h/2.0)*m2, lum + (h/2.0)*n2)
		l3 = self.dTdr(radius + (h/2.0), density + (h/2.0)*k2, temp + (h/2.0)*l2, mass + (h/2.0)*m2, lum + (h/2.0)*n2)
		m3 = self.dMdr(radius + (h/2.0), density + (h/2.0)*k2)
		n3 = self.dLdr(radius + (h/2.0), density + (h/2.0)*k2, temp + (h/2.0)*l2)
		o3 = self.dtaudr(density + (h/2.0)*k2, temp + (h/2.0)*l2)
		
		#increment based on the slope at the end of the interval using yn + h*k3
		k4 = self.dpdr(radius + h, density + h*k2, temp + h*l2, mass + h*m2, lum + h*n2)
		l4 = self.dTdr(radius + h, density + h*k2, temp + h*l2, mass + h*m2, lum + h*n2)
		m4 = self.dMdr(radius + h, density + h*k2)
		n4 = self.dLdr(radius + h, density + h*k2, temp + h*l2)
		o4 = self.dtaudr(density + h*k2, temp + h*l2)
	
		radius = radius + h #means radius_n+1
		density = density + (h/6.0)*(k1+2.0*k2+2.0*k3+k4) #means y_n+1 -- rk4 approximation of y(tn+1)
		temp = temp + (h/6.0)*(l1+2.0*l2+2.0*l3+l4)
		mass = mass + (h/6.0)*(m1+2.0*m2+2.0*m3+m4)
		lum = lum + (h/6.0)*(n1+2.0*n2+2.0*n3+n4)
		
		#populate the lists
		self.radius.append(radius)
		self.density.append(density)
		self.temp.append(temp)
		self.mass.append(mass)
		self.lum.append(lum)
		
		print "Radius:", radius, "\nDensity:", density, "\nTemp:", temp, "\nMass", mass, "\nLuminosity:", lum
		
	def OpacityLimit(self):
		dtau = (self.Kappa(self.density[-1]*self.temp[-1])*self.density[-1]**2)/abs(self.dpdr(self.radius[-1],self.density[-1],self.temp[-1],self.mass[-1],self.lum[-1])
		
		if dtau < 0.0001:
			OpacityReached = True
			
		else:
			OpacityReached = False
			
		if self.mass[-1] > (1.0e3)*ct.M_s:
			OpacityReached = True
			
		return OpacityReached
	
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
	def dpdr(self,radius,density,temp,mass,lum):
		return -((ct.G*mass*density)/(radius**2) + self.dPdT(density,temp)*self.dTdr(mass,density,temp,radius,lum))/self.dPdp(density,temp)
	
	#temp differential eqn
	def dTdr(self,radius,density,temp,mass,lum):
		return -min(self.dTdr_rad(density,lum,temp,radius),self.dTdr_conv(temp,mass,density,radius))
		
	def dTdr_rad(self,density,lum,temp,radius):
		return (3.0*self.Kappa(density,temp)*density*lum)/(16.0*np.pi*ct.a*ct.c*temp**3*radius**2)
		
	def dTdr_conv(self,temp,mass,density,radius):
		return (1.0 - (1.0/ct.gamma))*(temp*ct.G*mass*density)/(self.P*radius**2)
		
	#mass differential eqn
	def dMdr(self,radius,density):
		return 4.0*np.pi*radius**2*density
		
	#luminosity differential eqn
	def dLdr(self,radius,density,temp):
		return 4.0*np.pi*radius**2*density*self.Epsilon(density,temp)
		
	#optical depth differential eqn
	def dtaudr(self,density,temp):
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
			Kappa = ((1.0/self.KH(density,temp))+(1.0/max(Kes,self.Kff(density,temp))))**(-1.0)
			
		else:
			Kappa = ((1.0/self.KH(density,temp))+(1.0/min(Kes,self.Kff(density,temp))))**(-1.0)
			
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
		

for 
SingleStar(1000,6e4,1.0e8)	