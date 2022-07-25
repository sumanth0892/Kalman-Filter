import numpy as np
class KalmanFilter(object):
	def __init__(self,dim_x,dim_z):
		"""
		dim_x: Number of state variables
		dim_z: Number of measurement inputs
		"""
		self.dim_x = dim_x
		self.dim_z = dim_z

		self.x = np.zeros((dim_x,1))
		self.P = np.eye(dim_x) #Uncertainty Covariance
		self.Q = np.eye(dim_x) #Process Uncertainty
		self.u = 0
		self.B = np.zeros((dim_x,1))
		self.F = 0 #State transition matrix
		self.H = 0 #Measurement function
		self.R = np.eye(dim_z) #State uncertainty

		#Identity matrix
		self._I = np.eye(dim_x)

	def predict(self):
		self.x = self.F.dot(self.x) + self.B.dot(self.u)
		self.P = self.F.dot(self.P).dot(self.F.T) + self.Q

	def update(self, Z, R = None):
		"""
		Add a new measurement (Z) to the Kalman filter.
		Optionally provide R to override the measurement noise for this one call, otherwise use self.R
		"""
		if Z is None:
			return
		if R is None:
			R = self.R
		elif np.isscalar(R):
			R = np.eye(self.dim_z)*R

		#Error (residual) between measurement and prediction
		self.residual = Z - self.H.dot(self.x)

		#Project system uncertainty into measurement space
		self.S = self.H.dot(self.P).dot(self.H.T) + R

		#Map system uncertainty into Kalman gain
		self.K = self.P.dot(self.H.T).dot(np.linalg.inv(self.S))

		#Predict new x with the residual
		self.x += self.K.dot(self.residual)
		KH = self.K.dot(self.H)
		I_KH = self._I - KH
		self.P = (I_KH.dot(self.P.dot(I_KH.T)) +
				 self.K.dot(self.R.dot(self.K.T)))
		
		