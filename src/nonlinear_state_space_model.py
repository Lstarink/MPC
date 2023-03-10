import sympy as sm
import math as mt
import config

class StateSpaceModel:
    def __init__(self):
        x, y, z, x_dot, y_dot, z_dot = sm.Symbols("x, y, z, x_dot, y_dot, z_dot")

        # states
        self.states = sm.Matrix([x, y, z, x_dot, y_dot, z_dot])

        c1, c2, c3, c4, c5, c6, c7, c8 = sm.Symbols("c1, c2, c3, c4, c5, c6, c7, c8")
        # inputs
        self.inputs = sm.Matrix([c1, c2, c3, c4, c5, c6, c7, c8]).T

        # state equation
        self.f = StateSpaceModel.F(self)

    def F(self):
        f = sm.Matrix([0, 0, 0, 0, 0, 0])
        attachment_points = config.attachment_points
        for cable_attachment_point, cable_tension in zip(attachment_points, self.inputs):
            f += StateSpaceModel.Force(self, cable_tension, cable_attachment_point)
        return f

    def Force(self, cable_tension, cable_attachment_point):
        location = self.states[0:3]
        center_of_mass_to_attachment_point = cable_attachment_point - location
        unit_vector = center_of_mass_to_attachment_point/(sm.sqrt(center_of_mass_to_attachment_point.dot(center_of_mass_to_attachment_point)))
        force = unit_vector*cable_tension
        return force
