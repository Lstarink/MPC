import sympy as sm
import math as mt
import config


class StateSpaceModel:
    def __init__(self):
        x, y, z, x_dot, y_dot, z_dot = sm.symbols("x, y, z, x_dot, y_dot, z_dot")

        # states
        self.states = sm.Matrix([x, y, z, x_dot, y_dot, z_dot])

        c1, c2, c3, c4, c5, c6, c7, c8 = sm.symbols("c1, c2, c3, c4, c5, c6, c7, c8")
        # inputs
        self.inputs = sm.Matrix([c1, c2, c3, c4, c5, c6, c7, c8]).T

        # state equation
        self.f = StateSpaceModel.F(self)
        print(self.f.shape)

    def Evaluate(self, state_numerical, input_numerical):
        lambda_f = sm.utilities.lambdify([self.states, self.inputs], self.f)
        f_evaluated = lambda_f(state_numerical, input_numerical)
        return f_evaluated

    def F(self):
        f1 = sm.Matrix([self.states[3], self.states[4], self.states[5]])
        f2 = sm.Matrix([0, 0, -9.81])
        attachment_points = config.attachment_points
        for n, cable_tension in enumerate(self.inputs):
            f2 += StateSpaceModel.Force(self, cable_tension, attachment_points[n][0:3])

        f = sm.Matrix([f1,f2])
        return f

    def Force(self, cable_tension, cable_attachment_point):
        location = sm.Matrix([self.states[0], self.states[1], self.states[2]])
        center_of_mass_to_attachment_point = sm.Matrix([cable_attachment_point]).T - location
        unit_vector = center_of_mass_to_attachment_point/(sm.sqrt(center_of_mass_to_attachment_point.dot(center_of_mass_to_attachment_point)))
        force = unit_vector*cable_tension
        return force
