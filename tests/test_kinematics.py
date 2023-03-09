import unittest
import numpy as np
import math as mt
import kinematics

class TestKinematics(unittest.TestCase):
    def test_refB_to_refA_rotations(self):
        return 0
        # rB0 = np.array([[1], [0], [0]])
        # location0 = np.array([[0], [0], [0]])
        # angles0 = np.array([0, 0, 0])
        # rA0 = kinematics.refB_to_refA(angles0, location0, rB0)
        #
        # angles1 = np.array([mt.pi, 0, 0])
        # rA1 = kinematics.refB_to_refA(angles1, location0, rB0)
        #
        # angles2 = np.array([0, mt.pi, 0])
        # rA2 = kinematics.refB_to_refA(angles2, location0, rB0)
        #
        # angles3 = np.array([0, mt.pi/2, 0])
        # rA3 = kinematics.refB_to_refA(angles3, location0, rB0)
        #
        # angles4 = np.array([0, 0, mt.pi])
        # rA4 = kinematics.refB_to_refA(angles4, location0, rB0)
        #
        # angles5 = np.array([0, 0, mt.pi/2])
        # rA5 = kinematics.refB_to_refA(angles5, location0, rB0)
        #
        # angles6 = np.array([mt.pi, -mt.pi/2, mt.pi])
        # rA6 = kinematics.refB_to_refA(angles6, location0, rB0)
        #
        # floating_point_error = 1E-10
        #
        # dot0 = np.dot(np.transpose(rA0), rB0)
        # dot1 = np.dot(np.transpose(rA1), rB0)
        # dot2 = np.dot(np.transpose(rA2), rB0)
        # dot3 = np.dot(np.transpose(rA3), rB0)
        # dot4 = np.dot(np.transpose(rA4), rB0)
        # dot5 = np.dot(np.transpose(rA5), rB0)
        #
        # print(rA6)
        # self.assertLess(abs(dot0 - 1.0), floating_point_error)
        # self.assertLess(abs(dot1 - 1.0), floating_point_error)
        # self.assertLess(abs(dot2 + 1.0), floating_point_error)
        # self.assertLess(abs(dot3), floating_point_error)
        # self.assertLess(abs(dot4 + 1.0), floating_point_error)
        # self.assertLess(abs(dot5), floating_point_error)
        # # self.assertLess(np.linalg.norm(rA6-np.array([[0],[0],[-1]])),floating_point_error)
