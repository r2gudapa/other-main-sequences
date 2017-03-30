import matplotlib.pyplot as plt



x_ar = []
u1_ar = []
    
def rk4(n):

	x = 0.000001
	u1 = 1.0
	u2 = 0.0

	h = 0.05

	print "\n" * 40
	print "x ", "y"
	print x, u1
	
	while (x < 11):
		m1 = u2
		k1 = -(2/x)*u2 - u1**n #k1
		m2 = u2 + (h/2.)*k1
		x_2 = x + (h/2.)
		u1_2 = u1 + (h/2.)*m1
		u2_2 = m2
		
		k2 = -(2/x_2)*u2_2 - u1_2**n #k2
		m3 = u2 + (h/2.)*k2
		x_3 = x + (h/2.)
		u1_3 = u1 + (h/2.)*m2
		u2_3 = m3
		
		k3 = -(2/x_3)*u2_3 - u1_3**n #k4
		m4 = u2 + h*k3
		x_4 = x + h
		u1_4 = u1 + h*m3
		u2_4 = m4
		
		k4 = -(2/x_4)*u2_4 - u1_4**n #k4
		x = x + h
		u1 = u1 + (h/6.)*(m1 + (2. * m2) + (2. * m3) + m4)
		u2 = u2 + (h/6.)*(k1 + (2. * k2) + (2. * k3) + k4)
		x_ar.append(x)
		u1_ar.append(u1)
		print x, u1, u2	

for n in range(0,6):
        print "n = ", n
	rk4(n)	
	plt.plot(x_ar,u1_ar,label= n)
	x_ar = []
	u1_ar = []

axes = plt.gca()
axes.set_xlim(0,11)
axes.set_ylim(-1,1)
plt.legend(loc='best',bbox_to_anchor=(0.8,0.66),prop={'size':11})
plt.title("$\\theta$ vs. x for n = 0, 1, 2, 3, 4 and 5", fontsize="20")
plt.xlabel("x", fontsize="20")
plt.ylabel("$\\theta$", fontsize="20")
plt.grid()
plt.show()
	
	
