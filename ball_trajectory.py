import math
from numpy import random
def rk4(y,x,dx,f):
    """
    4th order Runge-Kutta for dy/dx
    y is the initial value of y
    x is the initial value of x
    dx is the time step
    f is a function that computes the gradient dy/dx
    """
    k1 = dx*f(y,x)
    k2 = dx*f(y+0.5*k1,x+0.5*dx)
    k3 = dx*f(y+0.5*k2,x+0.5*dx)
    k4 = dx*f(y+k3,x+dx)
    return y + (k1+2*k2+2*k3+k4)/6

def fx(x,t):
	return fx.vel

def fy(y,t):
	return fy.vel - 9.8*t

class Ball_2D(object):
	def __init__(self,x0,y0,velocity,theta_deg = 0,g = 9.8,noise = [0.0,0.0]):
		self.x = x0
		self.y = y0
		self.t = 0
		theta = math.radians(theta_deg)
		fx.vel = math.cos(theta)*velocity
		fy.vel = math.sin(theta)*velocity
		self.g = g
		self.noise = noise

	def step(self,dt):
		self.x = rk4(self.x,self.t,dt,fx)
		self.y = rk4(self.y,self.t,dt,fy)
		self.t += dt
		return (self.x + random.randn()+self.noise[0],
				self.y + random.randn()*self.noise[1])