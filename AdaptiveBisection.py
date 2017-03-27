import numpy as np
import time
import matplotlib.pyplot as plt
import constants as ct
from newStar1 import SingleStar

class FixDensity:

	def __init__(self,h,temp_c):
		self.h = h
		self.central_temp = temp_c
		print "StarA"
		self.starA = SingleStar(self.h,0.3e3,temp_c,0)
		print "StarB"
		self.starB = SingleStar(self.h,500.0e3,temp_c,0)
		print "StarC"
		self.starC = SingleStar(self.h,(0.3e3+500.0e3,0)/2.0,temp_c)
		
		self.BestStar = self.bisection(self.starA,self.starB,self.starC,0.001)
		
	def f(self,trialstar):
		
		RstarIndex = trailstar.Rstar
		
		Lum1 = trailstar.l[RstarIndex]
		Lum2 = (4.0*np.pi*ct.stfb*(trialstar[RstarIndex]**2.0)*(trialstar.t[RstarIndex]**4.0))
		
		return (Lum1 - Lum2)/np.sqrt(Lum1*Lum2)
		
	def bisection(starA,starB,starC,tol):
	
		starCrho = (starA.d[0] + starB.d[0])/2.0 
		
		while (starB.d[0]-starA.d[0])/2.0 > tol:
			if starC == 0:
				return starC
			
			elif f(starA)*f(starC) < 0:
				starB = starC
				
			else:
				starA = starC
				
			starCrho = (starA.d[0] + starB.d[0])/2.0 
			starC = SingleStar(self.h,starCrho,self.central_temp,0)
			print "New central density is:", starCrho
		return starC
		
FixDensity(1000.0,6.5e6)
		
		