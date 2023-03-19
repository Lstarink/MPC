import unittest
import numpy as np
import scipy as sp
import LQR_controller
import nonlinear_state_space_model

class TestLQR(unittest.TestCase):
    def test_LQR_only_stable_eigenvalues(self):
        dt = 0.1
        statespace = nonlinear_state_space_model.StateSpaceModel()
        statespace.Setdt(dt)

        A = statespace.Phi
        B = statespace.Gamma
        Q = np.identity(statespace.n)
        R = np.identity(statespace.m)

        controller = LQR_controller.LQR(A, B, Q, R)
        R_inv = sp.linalg.inv(R)

        cl_A = A - B@R_inv@B.T@controller.P
        [eig, eigv] = sp.linalg.eig(cl_A)
        for eigenvalue in eig:
            self.assertLess(abs(eigenvalue), 1.0)
    def test_LQR_zero_input_zero_output(self):
        dt = 0.1
        statespace = nonlinear_state_space_model.StateSpaceModel()
        statespace.Setdt(dt)

        A = statespace.Phi
        B = statespace.Gamma
        Q = np.identity(statespace.n)
        R = np.identity(statespace.m)

        controller = LQR_controller.LQR(A, B, Q, R)
        x = np.zeros(statespace.n)
        u = controller.Tick(x)

        self.assertIsNone(np.testing.assert_allclose(u, np.zeros(statespace.m)))
