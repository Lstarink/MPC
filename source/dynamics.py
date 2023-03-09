import numpy as np
import math as mt
import sympy as sm


class DynamicModel:
    def __init__(self):
        self.x = sm.symbols('x')
        self.y = sm.symbols('y')
        self.z = sm.symbols('z')

        self.x_dot = sm.symbols('x_dot')
        self.y_dot = sm.symbols('y_dot')
        self.z_dot = sm.symbols('z_dot')

        self.phi = sm.symbols('phi')
        self.psi = sm.symbols('psi')
        self.theta = sm.symbols('theta')

        self.phi_dot = sm.symbols('phi_dot')
        self.psi_dot = sm.symbols('psi_dot')
        self.theta_dot = sm.symbols('theta_dot')

        self.rotation_matrix = DynamicModel.RotationMatrix(self)


    def RotationMatrix(self):
        phi, psi, theta = sm.symbols("phi, psi, theta")

        C1 = sm.Matrix([[1, 0, 0],
                       [0, sm.cos(phi), -sm.sin(phi)],
                       [0, sm.sin(phi), sm.cos(phi)]])
        C2 = sm.Matrix([[sm.cos(psi), 0, sm.sin(psi)],
                       [0, 1, 0],
                       [-sm.sin(psi), 0, sm.cos(psi)]])
        C3 = sm.Matrix([[sm.cos(theta), -sm.sin(theta), 0],
                       [sm.sin(theta), sm.cos(theta), 0],
                       [0, 0, 1]])
        rotation_matrix = C3 @ C2 @ C1

        return rotation_matrix

    def UnitVector(self, fixed_point, point_on_cube_local):
        center_of_mass = sm.array([self.x, self.y, self.z])
        point_on_cube_global = center_of_mass + self.rotation_matrix @ point_on_cube_local

        point_on_cube_to_fixed_point =  (fixed_point-point_on_cube_global)
        unit_vector = point_on_cube_to_fixed_point/sm.sqrt(point_on_cube_to_fixed_point.dot(point_on_cube_to_fixed_point))
        return unit_vector

    def CalcForceAndMoment(self, fixed_point, point_on_cube_local, wire_tension):
        unit_vector = DynamicModel.UnitVector(self, fixed_point, point_on_cube_local)
        force = unit_vector*wire_tension

        r_center_of_mass_to_point_on_cube = self.rotation_matrix @ point_on_cube_local
        moment = r_center_of_mass_to_point_on_cube.cross(unit_vector*wire_tension)

        return force, moment



if __name__ == "__main__":
    my_model = DynamicModel()