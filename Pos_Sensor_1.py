import copy
import numpy as np
import matplotlib.pyplot as plt
from Kalman_Filter import KalmanFilter

class PosSensor1(object):
	def __init__(self,pos = [0,0],vel = (0,0), noise_scale = 1.):
		self.vel = vel
		self.noise_scale = noise_scale
		self.pos = copy.deepcopy(pos)

	def read(self):
		self.pos[0] += self.vel[0]
		self.pos[1] += self.vel[1]
		return [self.pos[0] + np.random.randn()*self.noise_scale,
				self.pos[1] + np.random.randn()*self.noise_scale]

#A small test
pos = [4,3]
s = PosSensor1(pos,(2,1),1)
for i in range(50):
	pos = s.read()
	plt.scatter(pos[0],pos[1])
plt.show()

f1 = KalmanFilter(dim_x = 4, dim_z = 2)
dt = 1
f1.F = np.array([[1,dt,0,0],
				[0,1,0,0],
				[0,0,1,dt],
				[0,0,0,1]])
