import unittest
import nonlinear_state_space_model

class TestStateSpace(unittest.TestCase):
    def test(self):
        state_space = nonlinear_state_space_model.StateSpaceModel()
        self.assertEqual(0,0)
