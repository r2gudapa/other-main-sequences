import constants as ct
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

Kes = 0.02*(1.0 + ct.X) #m^2/kg/m^3 #electron scattering opacity
X_cno = 0.03*ct.X #fraction of H going under CNO
mu = (2.0*ct.X + 0.75*ct.Y + 0.5*ct.Z)**(-1) #mean molecular weight for 100% ionization

#boundary conditions
r0 = 0.000001 #m -- initial radius

#rk constants
c = [0.0,0.0,0.2,0.3,0.8,8.0/9.0,1.0,1.0]

a2 = [0.0,0.2]
a3 = [0.0,0.075,0.225]
a4 = [0.0,44.0/45.0,-56.0/15.0,32.0/9.0]
a5 = [0.0,19372.0/6561.0,-25360.0/2187.0,64448.0/6561.0,-212.0/729.0]
a6 = [0.0,9017.0/3168.0,-355.0/33.0,46732.0/5247.0,49.0/176.0,-5103.0/18656.0]

b = [0.0,35.0/384.0,0.0,500.0/1113.0,125.0/192.0,-2187.0/6784.0,11.0/84.0]
bstar = [0.0,5179.0/57600.0,0.0,7571.0/16695.0,393.0/640.0,-92097.0/339200.0,1.0/40.0]

S=0.90 #Tolerance error #2
class SingleStar:
	
	def __init__(self, dr, rho_central, T_central):
		print 'init begin'	
		#lists for things we need to plot, these will be filled as rk4 integrates the functions
		#to begin with these only include the initial conditions and as rk4 finds more values, it should append to these
		self.h = dr  #step size -- not constant since we are planning to use adaptive rk4, this will change
		self.radius = [r0]
		self.density = [rho_central] 
		self.temp = [T_central] 
		self.mass = [(4.0/3.0)*np.pi*self.radius[0]**3*self.density[0]] 
		self.lum = [(4.0/3.0)*np.pi*r0**3*self.density[0]*self.Epsilon(self.density[0],self.temp[0])]
     
        
		self.error_density = 0.0 			        
		self.errtol = 100
		self.test = self.CreateStar() #test the class for one star
		self.plotting()	
	#use this function to output stuff
	def CreateStar(self):
		print 'createstar begin'
		done = False
		while done == False:
			self.rk4(self.radius[-1],self.density[-1],self.temp[-1],self.mass[-1],self.lum[-1], self.h)
			done = self.OptDepthLimit()

	#create a function for rk4
	def rk4(self,radius,density,temp,mass,lum,h):
		#Thank you Wikipedia!!
		#radius is the independent variable,
		#the rest are the dependent variables
		#h is step size
		print 'Step Size=', h
#increment based on the slope at the beginning of the interval using yn
		k1 = h*self.dpdr(radius,density,temp,mass,lum)	
		l1 = h*self.dTdr(radius,density,temp,mass,lum)
		m1 = h*self.dMdr(radius,density)
		n1 = h*self.dLdr(radius,density,temp)
		o1 = h*self.dtaudr(density,temp)
		
		#increment based on the slope at the midpoint of the interval using
		k2 = h*self.dpdr(radius + c[2]*h, density + a2[1]*k1, temp + a2[1]*l1, mass + a2[1]*m1, lum + a2[1]*n1)
		l2 = h*self.dTdr(radius + c[2]*h, density + a2[1]*k1, temp + a2[1]*l1, mass + a2[1]*m1, lum + a2[1]*n1)
		m2 = h*self.dMdr(radius + c[2]*h, density + a2[1]*k1)
		n2 = h*self.dLdr(radius + c[2]*h, density + a2[1]*k1, temp + a2[1]*l1)
		o2 = h*self.dtaudr(density + a2[1]*k1, temp + a2[1]*l1)
		
		#increment based on the slope at the midpoint of the interval using 
		k3 = h*self.dpdr(radius + c[3]*h, density + a3[1]*k1 + a3[2]*k2, temp + a3[1]*l1 + a3[2]*l2, mass + a3[1]*m1 + a3[2]*m2, lum + a3[1]*n1 + a3[2]*n2)
		l3 = h*self.dTdr(radius + c[3]*h, density + a3[1]*k1 + a3[2]*k2, temp + a3[1]*l1 + a3[2]*l2, mass + a3[1]*m1 + a3[2]*m2, lum + a3[1]*n1 + a3[2]*n2)
		m3 = h*self.dMdr(radius + c[3]*h, density + a3[1]*k1 + a3[2]*k2)
		n3 = h*self.dLdr(radius + c[3]*h, density + a3[1]*k1 + a3[2]*k2, temp + a3[1]*l1 + a3[2]*l2)
		o3 = h*self.dtaudr(density + a3[1]*k1 + a3[2]*k2, temp + a3[1]*l1 + a3[2]*l2)
		
		#increment based on the slope at the end of the interval using 
		k4 = h*self.dpdr(radius + c[4]*h, density + a4[1]*k1 + a4[2]*k2 + a4[3]*k3, temp + a4[1]*l1 + a4[2]*l2 + a4[3]*l3, mass + a4[1]*m1 + a4[2]*m2 + a4[3]*m3, lum + a4[1]*n1 + a4[2]*n2 + a4[3]*n3)
		l4 = h*self.dTdr(radius + c[4]*h, density + a4[1]*k1 + a4[2]*k2 + a4[3]*k3, temp + a4[1]*l1 + a4[2]*l2 + a4[3]*l3, mass + a4[1]*m1 + a4[2]*m2 + a4[3]*m3, lum + a4[1]*n1 + a4[2]*n2 + a4[3]*n3)
		m4 = h*self.dMdr(radius + c[4]*h, density + a4[1]*k1 + a4[2]*k2 + a4[3]*k3)
		n4 = h*self.dLdr(radius + c[4]*h, density + a4[1]*k1 + a4[2]*k2 + a4[3]*k3, temp + a4[1]*l1 + a4[2]*l2 + a4[3]*l3)
		o4 = h*self.dtaudr(density + a4[1]*k1 + a4[2]*k2 + a4[3]*k3, temp + a4[1]*l1 + a4[2]*l2 + a4[3]*l3)
		
		#fifth order
		k5 = h*self.dpdr(radius + c[5]*h, density + a5[1]*k1 + a5[2]*k2 + a5[3]*k3 + a5[4]*k4, temp + a5[1]*l1 + a5[2]*l2 + a5[3]*l3 + a5[4]*l4, mass + a5[1]*m1 + a5[2]*m2 + a5[3]*m3 + a5[4]*m4, lum + a5[1]*n1 + a5[2]*n2 + a5[3]*n3 + a5[4]*n4)
		l5 = h*self.dTdr(radius + c[5]*h, density + a5[1]*k1 + a5[2]*k2 + a5[3]*k3 + a5[4]*k4, temp + a5[1]*l1 + a5[2]*l2 + a5[3]*l3 + a5[4]*l4, mass + a5[1]*m1 + a5[2]*m2 + a5[3]*m3 + a5[4]*m4, lum + a5[1]*n1 + a5[2]*n2 + a5[3]*n3 + a5[4]*n4)
		m5 = h*self.dMdr(radius + c[5]*h, density + a5[1]*k1 + a5[2]*k2 + a5[3]*k3 + a5[4]*k4)
		n5 = h*self.dLdr(radius + c[5]*h, density + a5[1]*k1 + a5[2]*k2 + a5[3]*k3 + a5[4]*k4, temp + a5[1]*l1 + a5[2]*l2 + a5[3]*l3 + a5[4]*l4)
		o5 = h*self.dtaudr(density + a5[1]*k1 + a5[2]*k2 + a5[3]*k3 + a5[4]*k4, temp + a5[1]*l1 + a5[2]*l2 + a5[3]*l3 + a5[4]*l4)
		
		#sixth order 
		k6 = h*self.dpdr(radius + c[6]*h, density + a6[1]*k1 + a6[2]*k2 + a6[3]*k3 + a6[4]*k4 + a6[5]*k5, temp + a6[1]*l1 + a6[2]*l2 + a6[3]*l3 + a6[4]*l4 + a6[5]*l5, mass + a6[1]*m1 + a6[2]*m2 + a6[3]*m3 + a6[4]*m4 + a6[5]*m5, lum + a6[1]*n1 + a6[2]*n2 + a6[3]*n3 + a6[4]*n4 + a6[5]*n5)
		l6 = h*self.dTdr(radius + c[6]*h, density + a6[1]*k1 + a6[2]*k2 + a6[3]*k3 + a6[4]*k4 + a6[5]*k5, temp + a6[1]*l1 + a6[2]*l2 + a6[3]*l3 + a6[4]*l4 + a6[5]*l5, mass + a6[1]*m1 + a6[2]*m2 + a6[3]*m3 + a6[4]*m4 + a6[5]*m5, lum + a6[1]*n1 + a6[2]*n2 + a6[3]*n3 + a6[4]*n4 + a6[5]*n5)
		m6 = h*self.dMdr(radius + c[6]*h, density + a6[1]*k1 + a6[2]*k2 + a6[3]*k3 + a6[4]*k4 + a6[5]*k5)
		n6 = h*self.dLdr(radius + c[6]*h, density + a6[1]*k1 + a6[2]*k2 + a6[3]*k3 + a6[4]*k4 + a6[5]*k5, temp + a6[1]*l1 + a6[2]*l2 + a6[3]*l3 + a6[4]*l4 + a6[5]*l5)
		o6 = h*self.dtaudr(density + a6[1]*k1 + a6[2]*k2 + a6[3]*k3 + a6[4]*k4 + a6[5]*k5, temp + a6[1]*l1 + a6[2]*l2 + a6[3]*l3 + a6[4]*l4 + a6[5]*l5)
		
		#non star values
		radius = radius + h #means radius_n+1
		density = density + b[1]*k1 + b[2]*k2 + b[3]*k3 + b[4]*k4 + b[5]*k5 + b[6]*k6 #means y_n+1 -- rk4 approximation of y(tn+1)
		temp = temp + b[1]*l1 + b[2]*l2 + b[3]*l3 + b[4]*l4 + b[5]*l5 + b[6]*l6
		mass = mass + b[1]*m1 + b[2]*m2 + b[3]*m3 + b[4]*m4 + b[5]*m5 + b[6]*m6
		lum = lum + b[1]*n1 + b[2]*n2 + b[3]*n3 + b[4]*n4 + b[5]*n5 + b[6]*n6
		
		#star values
		#radiusstar = radius + h #means radius_n+1
		densitystar = density + bstar[1]*k1 + bstar[2]*k2 + bstar[3]*k3 + bstar[4]*k4 + bstar[5]*k5 + bstar[6]*k6 #means y_n+1 -- rk4 approximation of y(tn+1)
		tempstar = temp + bstar[1]*l1 + bstar[2]*l2 + bstar[3]*l3 + bstar[4]*l4 + bstar[5]*l5 + bstar[6]*l6
		massstar = mass + bstar[1]*m1 + bstar[2]*m2 + bstar[3]*m3 + bstar[4]*m4 + bstar[5]*m5 + bstar[6]*m6
		lumstar = lum + bstar[1]*n1 + bstar[2]*n2 + bstar[3]*n3 + bstar[4]*n4 + bstar[5]*n5 + bstar[6]*n6
				
		#populate the lists
		self.radius.append(radius)
		self.density.append(density)
		self.temp.append(temp)
		self.mass.append(mass)
		self.lum.append(lum)
      
        
#		#Calculate Deltas and scales (in nr.com book equations)
		deltaden=abs(density-densitystar)
		scaleden=self.errtol*h*max(self.density[-1],self.density[-2])    
		deltatemp=abs(temp-tempstar)
		scaletemp=self.errtol*h*max(self.temp[-1],self.temp[-2])
		deltamass=abs(mass-massstar)
		scalemass=self.errtol*h*max(self.mass[-1],self.mass[-2])
		deltalum=abs(lum-lumstar)
		scalelum=self.errtol*h*max(self.lum[-1],self.lum[-2])  
		print deltaden/scaleden        
#		#Calculate error (Mean of all deltas/scales in nr.com book equations)
		error=np.sqrt((1.0/5.0)*(deltaden/scaleden)**2+(deltatemp/scaletemp)**2+(deltamass/scalemass)**2+(deltalum/scalelum)**2)
        
		print "error =", error
		self.h=(S*h)*(1.0/error)**(1.0/5.0)
#		print "Radius:", radius, "\nDensity:", density, "\nTemp:", temp, "\nMass", mass, "\nLuminosity:", lum

	#Optical depth limit checker
	def OptDepthLimit(self):
		dtau = (self.Kappa(self.density[-1],self.temp[-1])*self.density[-1]**2)/abs(self.dpdr(self.radius[-1],self.density[-1],self.temp[-1],self.mass[-1],self.lum[-1]))
		#print dtau
		if dtau < 1.0e14:
			OptDepthReached = True
			
		elif self.mass[-1] > (1.0e3)*ct.M_s:
			OptDepthReached = True
			
		else:
			OptDepthReached = False
			
		return OptDepthReached
	
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
	def P(self,density,temp):
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
		return ((1.0 - (1.0/ct.gamma))*(temp*ct.G*mass*density))/(self.P(density,temp)*radius**2)
		
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
		return (2.5e-32)*(ct.Z/0.02)*(density/1e3)**(0.5)*temp**9.0
	
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
		
	def plotting(self):
		plt.plot(self.radius, self.density)
		plt.ylabel('Density')
		plt.show()
		plt.plot(self.radius, self.temp)
		plt.ylabel('Temperature')
		plt.show()
		plt.plot(self.radius, self.mass)
		plt.ylabel('Mass')
		plt.show()
		plt.plot(self.radius, self.lum)
		plt.ylabel('Luminosity')
		plt.show()
#       plt.plot(self.radius, self.Tau)
#       plt.show()

SingleStar(1000.0,6.0e4,1.0e8)	