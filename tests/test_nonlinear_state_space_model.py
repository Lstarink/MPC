import unittest
import numpy as np
import math as mt
import nonlinear_state_space_model

class TestStateSpace(unittest.TestCase):
    def test(self):
        state_space = nonlinear_state_space_model.StateSpaceModel()
        f_evaluated1 = state_space.Evaluate(np.zeros(6), np.zeros(8))
        f_evaluated2 = state_space.Evaluate(np.zeros(6), np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
        f_evaluated3 = state_space.Evaluate(np.array([1, 0, 0, 0, 0, 0]), np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
        print(f_evaluated3)

        # zero input zero output
        self.assertEqual(np.linalg.norm(f_evaluated1), 9.81)
        # pull 4 strings located at the positive x side to generate acceleration in x direction and nothing else
        self.assertEqual(f_evaluated2[3], 4/mt.sqrt(3))
        # pull 4 strings located at the positive x side, with the mass in the same plane, so no acceleration
        self.assertEqual(np.linalg.norm(f_evaluated3 - f_evaluated1), 0)

