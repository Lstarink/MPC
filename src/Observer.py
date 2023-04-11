import numpy as np
import nonlinear_state_space_model as nlss
import scipy as sp

class Observer:
    def __init__(self, dt, u_eq, x_eq, poles):
        self.statespace = nlss.StateSpaceModel(dt=dt)
        self.statespace.Linearize(u_eq=u_eq, x_eq=x_eq)
        self.Phi = self.statespace.Phi
        self.Gamma = self.statespace.Gamma
        self.PhiLarge, self.GammaLarge = self.PhiAppend()
        self.C = np.concatenate([np.identity(6), np.identity(6)]).T
        self.L = sp.signal.place_poles(self.PhiLarge.T, self.C.T, poles).gain_matrix

        self.xn_hat = np.zeros(12)

    def PhiAppend(self):
        PhiAppend = np.zeros([12, 12])
        GammaAppend = np.zeros([12, 8])
        for i in range(12):
            for j in range(12):
                if (i < 6) and (j < 6):
                    PhiAppend[i, j] = self.Phi[i,j]

        # PhiAppend[6, 6] = 1
        PhiAppend[7, 7] = 1
        # PhiAppend[8, 8] = 1

        print(PhiAppend)
        print(self.Phi)
        for i in range(12):
            for j in range(8):
                if i < 6:
                    GammaAppend[i, j] = self.Gamma[i, j]

        print(GammaAppend)
        print(self.Gamma)
        # input("holdit")
        return PhiAppend, GammaAppend

    def Tick(self, yn, un):
        y_hat = self.C@self.xn_hat
        xplus_hat = self.PhiLarge@self.xn_hat+ self.GammaLarge@un + self.L.T@(yn - y_hat)
        self.xn_hat = xplus_hat
        return y_hat, self.xn_hat[6:12]

    def InitState(self, x0, d0):
        self.xn_hat[0:6] = x0
        self.xn_hat[6:12] = d0
        print(self.xn_hat)
