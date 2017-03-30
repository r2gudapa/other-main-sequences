import numpy as np
import time
import matplotlib.pyplot as plt
import constants as ct
from RegStar import SingleStar

start=time.time()

class FixDensity:

	def __init__(self,h,temp_c):
		self.h = h
		self.central_temp = temp_c
#		print "StarA"
		self.starA = SingleStar(self.h,0.3e3,temp_c,1)
#		print "StarB"
		self.starB = SingleStar(self.h,500.0e3,temp_c,1)
#		print "StarC"
		self.starC = SingleStar(self.h,(0.3e3+500.0e3)/2.0,temp_c,1)
		
		self.BestStar = self.bisection(self.starA,self.starB,self.starC,0.01)
		
	def f(self,trialstar):
		
		RstarIndex = trialstar.Rstar
		
		Lum1 = trialstar.l[RstarIndex]
		Lum2 = (4.0*np.pi*ct.stfb*(trialstar.r[RstarIndex]**2.0)*(trialstar.t[RstarIndex]**4.0))
		
		return (Lum1 - Lum2)/np.sqrt(Lum1*Lum2)
		
	def bisection(self,starA,starB,starC,tol):
	
		starCrho = (starA.d[0] + starB.d[0])/2.0 
#		print "star A=",starA.d[0], "; Star B=", starB.d[0]
		
		while (starB.d[0]-starA.d[0])/2.0 > tol:
#			print "---------------------------------"            
			if starC == 0:
				return starC
			
			elif self.f(starA)*self.f(starC) < 0:
				starB = starC
				
			else:
				starA = starC
                
				
			starCrho = (starA.d[0] + starB.d[0])/2.0 
              
			starC = SingleStar(self.h,starCrho,self.central_temp,1)
#			print "New central density is:", starCrho
#		print "star A=",starA.d[0], "; Star B=", starB.d[0]
		starCrho=max(starA.d[0], starB.d[0])            
		starC=SingleStar(self.h,starCrho,self.central_temp,0)            
		return starC
		
#FixDensity(1000.0,1.5e7)
		


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MAIN SEQUENCE FILE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class MainSequence:
	def __init__(self,NumStars,minTc,maxTc):
		self.NumStars = NumStars
		self.minTc = minTc
		self.maxTc =maxTc
		self.CreateMS()        
	
	def onestar(self,temp_c):
		star = FixDensity(1000.0,temp_c).BestStar
		return star
	
	def STemp(self,x):
		return ((x.l[x.Rstar])/(4.0*np.pi*ct.stfb*x.r[x.Rstar]**2.0))**(1.0/4.0)
		
	def CreateMS(self):
		tempCs = np.linspace(self.minTc,self.maxTc,self.NumStars)
		surfaceTemps = []
		starLums = []
		
#		pool = mp.Pool(mp.cpu_count())
#		MS = pool.map(self.onestar,tempCs)
	
		for i in range(len(tempCs)):	
			print "Star #", i+1                
			MS = self.onestar(tempCs[i])	            
			surfaceTemps.append(self.STemp(MS))
			starLums.append(MS.l[MS.Rstar])
        
		
			plt.scatter(np.log10(surfaceTemps[i]),np.log10(starLums[i]/ct.L_s))
            
		plt.gca().invert_xaxis()        
		plt.xlabel('log Temperature (K)')  
		plt.ylabel('log Luminosity (Watt)') 
		plt.savefig('MS.jpg', dpi=1000)         
		plt.show()
		

		
MainSequence(25,10**6.6,10**7.4)
		
print "It took:", (time.time()-start)//60, " minutes and", (time.time()-start)%60 ,"seconds to run?"