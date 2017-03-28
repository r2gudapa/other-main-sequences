from multiprocessing import Process, Queue
import numpy as np
from AdaptiveBisection import FixDensity
import constants as ct
import time

class MainSequence:
	def __init__(self,NumStars,minTc,maxTc):
		self.NumStars = NumStars
		self.minTc = minTc
		self.maxTc =maxTc
	
	def onestar(self,temp_c):
		star = FixDensity(1000.0,temp_c).BestStar
		return star
	
	def STemp(self,x):
		return ((x.l[x.Rstar])/(4.0*np.pi*ct.stfb*x.r[x.Rstar]**2.0))**(1.0/4.0)
		
	def CreateMS(self):
		tempCs = np.linspace(self.minTc,self.maxTc,self.NumStars)
		surfaceTemps = []
		starLums = []
		
		pool = mp.Pool(mp.cpu_count())
		MS = pool.map(self.onestar,tempCs)
		
		surfaceTemps = map(self.STemp(MS))
		starLums = map(MS.l[MS.Rstar])
		
		plt.scatter(surfaceTemps,starLums/ct.L_s)
		plt.show()
		
MainSequence(100,10**6.6,10**7.6)
		
		
		
	
