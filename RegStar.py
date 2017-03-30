import constants as ct
import numpy as np
import matplotlib.pyplot as plt
import time
mu = (2.0*ct.X + 0.75*ct.Y + 0.5*ct.Z)**(-1.0)
r0 = 0.001 #m
S=1.0 #error tolerance


class SingleStar:
    
    def __init__(self, dr, rho_c, temp_c,plotmode):
        self.plotmode=plotmode
        self.dr = dr
        self.Rstar = 0.0
        
        #lists to hold the values from rk4
        #stellar structurs variables
        self.r = [r0]
        self.d = [rho_c]
        self.t = [temp_c]
        #self.m = [0.0]
        self.m = [(4.0/3.0)*np.pi*(r0**3.0)*rho_c]
        
        self.l = [self.m[0]*self.epsilon(rho_c,temp_c)]
        self.tau = [self.Kappa(rho_c,temp_c)*rho_c]
        
        #other variables
        self.k = [self.Kappa(rho_c,temp_c)]
        self.p = [self.P(rho_c,temp_c)]
        self.dLdr_list = [0.0]
        self.dlogPdlogT = [(self.t[-1]/self.p[-1])*(self.dPdr(self.r[-1],self.m[-1],self.d[-1])/self.dTdr(self.r[-1],self.m[-1],self.d[-1],self.t[-1],self.l[-1]))]
        self.drray = [dr] 
        self.dtaurray =[]        
        
        self.test = self.CreateStar()
        #self.test =self.test1()
        self.RStar = self.RadStar()
        self.plot = self.Plots(plotmode)
        
   
        
    ###########______Define all Equations______###########
    
    #Pressures
    def P(self,density,temp):
    
        P_deg = (((3.0*np.pi**2.0)**(2.0/3.0))*(ct.hbar**(2.0))*((density/ct.m_p)**(5.0/3.0)))/(5.0*ct.m_e)
        
        P_ig = (density*ct.k*temp)/(mu*ct.m_p)
        
        P_rad = (1.0/3.0)*ct.a*(temp**4.0)
        
        return P_deg + P_ig + P_rad
        
    #Pressure differentials
    def dPdp(self,density,temp):
        
        dPdp_deg = (((3.0*np.pi**2.0)**(2.0/3.0))*(ct.hbar**2.0)*((density/ct.m_p)**(2.0/3.0)))/(3.0*ct.m_p*ct.m_e)
        
        dPdp_ig = (ct.k*temp)/(mu*ct.m_p)
        
        return dPdp_deg + dPdp_ig
        
    def dPdT(self,density,temp):
        
        dPdT_ig = (density*ct.k)/(mu*ct.m_p)
        
        dPdT_rad = (4.0/3.0)*ct.a*(temp**3.0)

        return dPdT_ig + dPdT_rad
    
    #Energy generation
    def epsilon(self,density,temp):
        
        epp = (1.07e-7)*(density/1.0e5)*(ct.X**2.0)*((temp/1.0e6)**4.0)
        
        ecno = (8.24e-26)*(density/1.0e5)*0.03*(ct.X**2.0)*((temp/1.0e6)**19.9)
        
        return epp + ecno
        
    #Opacity
    def Kappa(self,density,temp):
        
        Kes = 0.02*(1.0+ct.X)
        
        Kff = (1.0e24)*(ct.Z+0.0001)*((density/1.0e3)**0.7)*temp**(-3.5)

        Khminus = (2.5e-32)*(ct.Z/0.02)*((density/1.0e3)**0.5)*(temp**9.0)
        
        return ((1.0/Khminus) + (1.0/max(Kes,Kff)))**(-1.0)
        
    #Stellar Structure ODEs
    def dpdr(self,radius,mass,density,temp,lum):
    
        return -((ct.G*mass*density/(radius**2.0)) + self.dPdT(density,temp)*self.dTdr(radius,mass,density,temp,lum))/(self.dPdp(density,temp))
        
    def dTdr(self,radius,mass,density,temp,lum):
    
        dTdr_rad = (3.0*self.Kappa(density,temp)*density*lum)/(16.0*np.pi*ct.a*ct.c*(temp**3.0)*(radius**2.0))
        
        dTdr_conv = (1.0 - (1.0/ct.gamma))*(temp/self.P(density,temp))*((ct.G*mass*density)/(radius**2.0))
        
        return - min(dTdr_rad,dTdr_conv)
        
    def dMdr(self,radius,density):
        
        return 4.0*np.pi*(radius**2.0)*density
    
    def dLdr(self,radius,density,temp):
        
        return self.dMdr(radius,density)*self.epsilon(density,temp)
        
    def dtaudr(self,density,temp):
        
        return self.Kappa(density,temp)*density
        
    #delta(tau) for optical depth limit
    def dtau(self,radius,mass,density,temp,lum):
        
        return (self.Kappa(density,temp)*(density**2.0))/(abs(self.dpdr(radius,mass,density,temp,lum)))
  
    def dPdr(self,radius,mass,density):
        
        return -(ct.G*mass*density/(radius**2.0))
        
    def rk4(self,y,r,h,f):
        '''
        y: current values for dependent variables
        r: radius, the independent variables
        h: step-size
        f:function array to be integrated
        '''
        k1 = h*f(y,r)
        k2 = h*f(y + 0.5*k1,r + 0.5*h)
        k3 = h*f(y + 0.5*k2,r + 0.5*h)
        k4 = h*f(y + k3,r + h)
        
        return y + (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0, r + h 
    
    def adaptivestepsize(self, h):
        #Calculate Deltas and scales
#        deltaden=abs(self.d[-1]-self.d[-2])
#        scaleden=self.d[-1]    
#        deltatemp=abs(self.t[-1]-self.t[-2])
#        scaletemp=self.t[-1]
#        deltamass=abs(self.m[-1]-self.m[-2])
#        scalemass=self.m[-1]
#        deltalum=abs(self.l[-1]-self.l[-2])
#        scalelum=self.l[-1]
#        #print deltaden/scaleden        
#        #Calculate error (Mean of all deltas/scales in nr.com book equations)
#        error=np.sqrt((1.0/4.0)*(deltaden/scaleden)**2+(deltatemp/scaletemp)**2+(deltamass/scalemass)**2+(deltalum/scalelum)**2)
#        print "density error=",scaleden/deltaden        
#        if self.dr*error<1.0e-2:
#                self.dr=self.dr*S
#        else:
#                self.dr=self.dr/S                    
#         self.dr=max(min(S*h*((1.0/(scaleden/deltaden))**(1.0/1.0)),100000), 1000)
#         self.drray.append(self.dr)   
         if self.t[-1]<50.0e3:
                  self.dr=(0.00001*self.r[-1])+1000
         else:
                  self.dr=(0.001*self.r[-1])+1000              
#         print 'current step size=', self.dr


    def f(self,dep_var,r):
    
        radius = r
        density = dep_var[0]
        temp = dep_var[1]
        mass = dep_var[2]
        lum = dep_var[3]
        tau = dep_var[4]

        rho = self.dpdr(radius,mass,density,temp,lum)
        T = self.dTdr(radius,mass,density,temp,lum)
        M = self.dMdr(radius,density)
        L = self.dLdr(radius,density,temp)
        tau = self.dtaudr(density,temp)
        
        return np.array([rho,T,M,L,tau],float)
    
    #The following three functions as for the assignment 4 question to test whether rk4 is doing what it's supposed to! (And it is ^_^)
    def dpdrtest(self,radius,density,mass):
        return -density*mass/radius**2.0
        
    def dMdrtest(self,radius,density):
        return 4.0*np.pi*radius**2.0*density
        
    def ftest(self,dep_var,r):
        radius = r
        density = dep_var[0]
        mass = dep_var[1]
        
        rho = self.dpdrtest(radius,density,mass)
        M = self.dMdrtest(radius,density)
        
        return np.array([rho,M],float)
        
    def OpticalDepthLimit(self):
        
        dtau = self.dtau(self.r[-1],self.m[-1],self.d[-1],self.t[-1],self.l[-1])
        self.dtaurray.append(dtau)        
        if dtau < 0.0001:
            return True
            
        elif self.m[-1] > (1.0e3)*ct.M_s:
            return True
            
        else:
#            print dtau
            return False
        
    def RadStar(self):
        
        self.Rstar = np.argmin(abs(self.tau[-1] - np.array(self.tau) - (2.0/3.0)))
        
        if self.tau[self.Rstar] == 0:
            self.Rstar = len(self.tau) - 1
            
#        print "Radius of the star is:", self.r[self.Rstar]
        return self.Rstar
    #Uncomment this if running the test with assignment 4 question!
    '''
    def test1(self):
        
        while self.r[-1] < 5.0:
            rk4_var = np.array([self.d[-1],self.m[-1]],float)
            new_rk4, new_radius = self.rk4(rk4_var,self.r[-1],self.dr,self.ftest)
            
            self.d.append(new_rk4[0])
            self.m.append(new_rk4[1])
            self.r.append(new_radius)
    '''    
    def CreateStar(self):
        start = time.time()    
        done = False
        while done == False:
            
            rk4_var = np.array([self.d[-1],self.t[-1],self.m[-1],self.l[-1],self.tau[-1]],float)
            
            new_rk4, new_radius = self.rk4(rk4_var,self.r[-1],self.dr,self.f)
            
            #print new_rk4, new_radius
            
            self.d.append(new_rk4[0])
            self.t.append(new_rk4[1])
            self.m.append(new_rk4[2])
            self.l.append(new_rk4[3])
            self.tau.append(new_rk4[4])
            self.r.append(new_radius)
            self.dLdr_list.append(self.dLdr(self.r[-1],self.d[-1],self.t[-1]))
            self.k.append(self.Kappa(self.d[-1],self.t[-1]))
            self.p.append(self.P(self.d[-1],self.t[-1]))
            self.dlogPdlogT.append((self.t[-1]/self.p[-1])*(self.dPdr(self.r[-1],self.m[-1],self.d[-1])/self.dTdr(self.r[-1],self.m[-1],self.d[-1],self.t[-1],self.l[-1])))
            self.adaptivestepsize(self.dr)

            done = self.OpticalDepthLimit()
            #print self.k[-1]

#        print "Surface Radius=",self.r[-1]     
#        print "Surface Density=",self.d[-1]
#        print "Surface Temp=",self.t[-1]  
#        print "Surface Mass=",self.m[-1]  
#        print "Surface Lum=",self.l[-1]  
#        print "It took:", (time.time()-start)//60, " minutes and", (time.time()-start)%60 ,"seconds to run?"          
            #done = True
        
    def Plots(self,plotmode):
        
        #convert to arrays for normalizations
        r = np.array(self.r)
        rho = np.array(self.d)
        temp = np.array(self.t)
        mass = np.array(self.m)
        lum = np.array(self.l)
        tau = np.array(self.tau)
        pressure = np.array(self.p)
        kappa = np.array(self.k)
        dLdr = np.array(self.dLdr_list)
            
        if plotmode==0:    
        #plot the data
            plt.figure(1)
            plt.grid()
            plt.legend(loc='best',bbox_to_anchor=(0.8,0.66),prop={'size':11})
            plt.plot(r/self.r[self.Rstar], rho/self.d[0], label='rho')
            plt.plot(r/self.r[self.Rstar], temp/self.t[0], label='temp')
            plt.plot(r/self.r[self.Rstar], mass/self.m[-1], label='Mass')
            plt.plot(r/self.r[self.Rstar], lum/self.l[-1], label='Lum')
            plt.title("Rho", fontsize=25)
            plt.savefig('Fig1.png', dpi=1000)        
            plt.show()
            '''    
            plt.figure(2)
            plt.grid()
            plt.plot(r/self.Rstar, temp/self.t[0], label='temp')
            plt.title("Temp", fontsize=25)
            plt.show()
        
            plt.figure(3)
            plt.grid()
            plt.plot(r/self.Rstar, mass/self.m[-1], label='Mass')
            plt.title("Mass", fontsize=25)
            plt.show()
        
            plt.figure(4)
            plt.grid()
            plt.plot(r/self.Rstar, lum/self.l[-1], label='Lum')
            plt.title("Lum", fontsize=25)
            plt.show()
            '''    
            plt.figure(9)
            plt.grid()
            plt.plot(r/self.r[self.Rstar], tau/self.tau[-1], label='Tau')
            plt.title("Tau", fontsize=25)
            plt.savefig('Tau.png', dpi=1000) 
            plt.show()
        
            plt.figure(5)
            plt.grid()
            plt.plot(r/self.r[self.Rstar], dLdr/(self.l[-1]/self.r[self.Rstar]), label='dL/dR')
            plt.title("dLdR", fontsize=25)
            plt.savefig('dLdR.png', dpi=1000)        
            plt.show()
        
            plt.figure(6)
            plt.grid()
            plt.legend(loc='best',bbox_to_anchor=(0.8,0.66),prop={'size':11})
            plt.plot(r/self.r[self.Rstar], pressure/self.p[0], label='Pressure')
            plt.title("Pressure", fontsize=25)
            plt.savefig('Pressure.png', dpi=1000)        
            plt.show()
        
            plt.figure(7)
            plt.grid()
            plt.plot(r/self.r[self.Rstar], np.log10(kappa), label='Opacity')    
            plt.title("Opacity", fontsize=25)
            plt.savefig('Opacity.png', dpi=1000)        
            plt.show()
        
            plt.figure(8)
            axes = plt.gca()
            #axes.set_xlim(0,11)
            axes.set_ylim(0,10)
            plt.grid()
            plt.plot(r/self.r[self.Rstar], self.dlogPdlogT, label='dlogP/dlogT')
            plt.title("dlogP/dlogT", fontsize=25)
            plt.savefig('dlogP-dlogT.png', dpi=1000)        
            plt.show()
   
 
#SingleStar(1000.0,0.5e3,1.5e7)		
#print "It took:", (time.time()-start)//60, " minutes and", (time.time()-start)%60 ,"seconds to run?"