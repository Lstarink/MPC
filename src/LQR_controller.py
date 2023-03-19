import numpy as np
import scipy as sp

class LQR:
    def __init__(self, A, B, Q, R):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.P = np.zeros(A.shape)
        LQR.DARE(self)

        self.Rinv = sp.linalg.inv(R)
    def DARE(self):
        self.P = sp.linalg.solve_discrete_are(self.A, self.B, self.Q, self.R)

    def Tick(self, state):
        u = - self.Rinv@self.B.T@self.P@state
        return u
