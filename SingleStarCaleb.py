import constants as ct
import numpy as np
import matplotlib.pyplot as plt

Kes = 0.02*(1.0 + ct.X) #m^2/kg/m^3 #electron scattering opacity
X_cno = 0.03*ct.X #fraction of H going under CNO
mu = (2.0*ct.X + 0.75*ct.Y + 0.5*ct.Z)**(-1) #mean molecular weight for 100% ionization

#boundary conditions
r0 = 0.000001 #m -- initial radius

class SingleStar:
	
	def __init__(self, dr, rho_central, T_central,plotmode):
		
		#lists for things we need to plot, these will be filled as rk4 integrates the functions
		#to begin with these only include the initial conditions and as rk4 finds more values, it should append to these
		self.dr = dr #step size -- not constant since we are planning to use adaptive rk4, this will change
		self.radius = [r0]
		self.density = [rho_central] 
		self.temp = [T_central] 
		self.mass = [(4.0/3.0)*np.pi*(self.radius[0]**3)*self.density[0]]
		self.lum = [(4.0/3.0)*np.pi*(r0**3)*self.density[0]*self.Epsilon(self.density[0],self.temp[0])]
		
		self.dLdr_list = [(self.dLdr(self.radius[0],self.density[0],self.temp[0]))]
		
		self.pressure = [(self.P(self.density[0],self.temp[0]))]
		self.pressureDeg = [(self.P_deg(self.density[0]))]
		self.pressureIG = [(self.P_ig(self.density[0],self.temp[0]))]
		self.pressureRad = [(self.P_rad(self.temp[0]))]
		
		#derivative of log P is dPdr/P, and dlogT is dTdr/T 
		#so dlogP/dlogT = dP/dT*(T/P)
		self.dlogPdlogT = [(self.dPdT(self.density[0],self.temp[0]))*(self.temp[0]/self.density[0])]
		
		self.kappa = [(self.Kappa(self.density[0],self.temp[0]))]

		self.test = self.CreateStar() #test the class for one star
		self.plot = self.Plots(plotmode)
		
	
	#use this function to output stuff
	def CreateStar(self):
	
		done = False
		while done == False:
			self.rk4(self.radius[-1],self.density[-1],self.temp[-1],self.mass[-1],self.lum[-1], self.dr)
			done = self.OptDepthLimit()
					
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
		
		self.dLdr_list.append(self.dLdr(self.radius[-1],self.density[-1],self.temp[-1]))
		
		self.pressure.append(self.P(self.density[-1],self.temp[-1]))
		self.pressureDeg.append(self.P_deg(self.density[-1]))
		self.pressureIG.append(self.P_ig(self.density[-1],self.temp[-1]))
		self.pressureRad.append(self.P_rad(self.temp[-1]))
		
		#derivative of log P is dPdr/P, and dlogT is dTdr/T 
		#so dlogP/dlogT = dP/dT*(T/P)
		self.dlogPdlogT.append(self.dPdT(self.density[-1],self.temp[-1])*(self.temp[-1]/self.density[-1]))
		
		self.kappa.append(self.Kappa(self.density[-1],self.temp[-1]))
		
		
		print "Radius:", radius, "\nDensity:", density, "\nTemp:", temp, "\nMass", mass, "\nLuminosity:", lum, "\n"	
	#Optical depth limit checker
	def OptDepthLimit(self):
		dtau = (self.Kappa(self.density[-1],self.temp[-1])*self.density[-1]**2)/abs(self.dpdr(self.radius[-1],self.density[-1],self.temp[-1],self.mass[-1],self.lum[-1]))
		
		if dtau < 0.0001:
			OptDepthReached = True
			
		elif self.mass[-1] > (1.0e3)*ct.M_s:
			OptDepthReached = True
			
		else:
			OptDepthReached = False
			print "\n", dtau
			
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
		return -(((ct.G*mass*density)/(radius**2)) + (self.dPdT(density,temp)*self.dTdr(mass,density,temp,radius,lum)))/self.dPdp(density,temp)
	
	#temp differential eqn
	def dTdr(self,radius,density,temp,mass,lum):
		return -min(self.dTdr_rad(density,lum,temp,radius),self.dTdr_conv(temp,mass,density,radius))
		
	def dTdr_rad(self,density,lum,temp,radius):
		return (3.0*self.Kappa(density,temp)*density*lum)/(16.0*np.pi*ct.a*ct.c*temp**3*radius**2)
		
	def dTdr_conv(self,temp,mass,density,radius):
		return (1.0 - (1.0/ct.gamma))*(temp*ct.G*mass*density)/(self.P(density,temp)*radius**2)
		
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
		return (1.0e24)*(ct.Z+0.0001)*(density/1.0e3)**(0.7)*temp**(-3.5)
	
	#hydrogen ion opacity 	
	def KH(self,density,temp):
		return (2.5e-32)*(ct.Z/0.02)*(density/1.0e3)**(0.5)*temp**9
	
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
		
	def Plots(self,plotmode):
	
		if plotmode == 0:
			plt.figure(1)
			plt.grid()
			#plt.legend(loc='best',bbox_to_anchor=(0.8,0.66),prop={'size':11})
			plt.plot(self.radius, self.density, label='rho')
			#plt.plot(self.radius, self.temp, label='temp')
			#plt.plot(self.radius, self.lum, label='Lum')
			#plt.plot(self.radius, self.mass, label='Mass')
			plt.title("Rho", fontsize=25)
			plt.show()
			
			plt.figure(2)
			plt.grid()
			plt.plot(self.radius, self.temp, label='temp')
			plt.title("Temp", fontsize=25)
			plt.show()

			plt.figure(3)
			plt.grid()
			plt.plot(self.radius, self.mass, label='Mass')
			plt.title("Mass", fontsize=25)
			plt.show()

			plt.figure(4)
			plt.grid()
			plt.plot(self.radius, self.lum, label='Lum')
			plt.title("Lum", fontsize=25)
			plt.show()
			
			plt.figure(5)
			plt.grid()
			plt.plot(self.radius, self.dLdr_list, label='dL/dR')
			plt.title("dLdR", fontsize=25)
			plt.show()
			
			plt.figure(6)
			plt.grid()
			plt.legend(loc='best',bbox_to_anchor=(0.8,0.66),prop={'size':11})
			plt.plot(self.radius, self.pressure, label='Pressure')
			plt.plot(self.radius, self.pressureDeg, label='PressureDeg')
			plt.plot(self.radius, self.pressureIG, label='PressureIG')
			plt.plot(self.radius, self.pressureRad, label='PressureRad')
			plt.title("Pressure", fontsize=25)
			plt.show()
			
			plt.figure(7)
			plt.grid()
			plt.plot(self.radius, self.kappa, label='Opacity')
			plt.yscale('log', nonposy='clip')
			plt.title("Opacity", fontsize=25)
			plt.show()
			
		if plotmode == 1: 

			plt.grid()
			plt.plot(self.radius, self.density, label='rho')
			plt.plot(self.radius, self.temp, label='temp')
			plt.plot(self.radius, self.mass, label='Mass')
			plt.plot(self.radius, self.lum, label='Lum')
			plt.legend()
			plt.title("Graph 1", fontsize=25)
			plt.xlabel('r(m)', fontsize=20)
			plt.ylabel('$y$', fontsize=20)
			plt.savefig('Test', dpi=1000)
			plt.show()
			
		if plotmode == 2:
			print "No plots displayed"


class Star_with_bisection:
    
    def __init__(self,dr,rho_central,plotmode):
        self.dr = dr
        self.rho_central = rho_central
        self.plotmode = plotmode
        self.star_start = SingleStar(1000.0,6.0e3,rho_central,plotmode)
        self.star_end = SingleStar(1000.0,500.0e3,rho_central,plotmode)
        self.star_bisec = SingleStar(1000.0,(6.0e3+500.0e3)/2.0,rho_central,plotmode)
        self.star_final = self.bisection(self.star_start, self.star_bisec, self.star_end, 0.001)

    def bisection_function(self,radius,temp,lum):
    	return (lum-(4.0*np.pi*ct.stfb*(radius)**2.0*(temp)**4.0))/((4.0*np.pi*ct.stfb*(radius)**2.0*(temp)**4.0*lum)**(1.0/2.0))

    def bisection(self,star_start,star_bisec,star_end,tol):
        while ((star_end.density[0] - star_start.density[0]) / 2.0) > tol:
            if self.bisection_function(self.star_bisec) == 0:
               return self.star_bisec
            if self.bisection_function(self.star_start) * self.bisection_function(self.star_bisec) < 0:
                self.star_end = self.star_bisec
            else:
                self.star_start = self.star_bisec
            self.star_bisec_density = (star_start.density[0] + self.star_end.density[0]) / 2.0
            self.star_bisec = SingleStar(self.dr,self.star_bisec_density,self.rho_central,self.p)
        return self.star_bisec
    
Star_with_bisection(1000.0,6.0e3,0)	





