import unittest
import sympy as sm
import dynamics
import math as mt


class TestDynamics(unittest.TestCase):
    def test_RotationMatrixOrthogonalProperty(self):
        phi, psi, theta = sm.symbols("phi, psi, theta")
        dynamic_model = dynamics.DynamicModel()
        floating_point_error =1E-10
        identity = sm.Matrix([[1, 0, 0], [0, 1, 0], [0, 0 , 1]])


        ata_symbolic = sm.transpose(dynamic_model.rotation_matrix) @ dynamic_model.rotation_matrix
        f = sm.utilities.lambdify([(phi, psi, theta)], ata_symbolic)
        ata = f((3, 4, 5))

        self.assertLess((ata-identity).norm(), floating_point_error)

    def test_RotationMatrixRotate(self):
        phi, psi, theta = sm.symbols("phi, psi, theta")
        dynamic_model = dynamics.DynamicModel()
        floating_point_error =1E-10
        f = sm.utilities.lambdify([(phi, psi, theta)], dynamic_model.rotation_matrix)

        e_x = sm.Matrix([1, 0, 0]).T
        e_y = sm.Matrix([0, 1, 0]).T
        e_z = sm.Matrix([0, 0, 1]).T

        rotation_matrix1 = f([mt.pi, 0, 0])


