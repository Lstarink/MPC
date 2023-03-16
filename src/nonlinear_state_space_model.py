import sympy as sm
import math as mt
import numpy as np
import config
import scipy as sp


class StateSpaceModel:
    def __init__(self):
        x, y, z, x_dot, y_dot, z_dot = sm.symbols("x, y, z, x_dot, y_dot, z_dot")

        # states
        self.statesym = sm.Matrix([x, y, z, x_dot, y_dot, z_dot])

        c1, c2, c3, c4, c5, c6, c7, c8 = sm.symbols("c1, c2, c3, c4, c5, c6, c7, c8")
        # inputs
        self.inputs = sm.Matrix([c1, c2, c3, c4, c5, c6, c7, c8]).T

        self.n = len(self.statesym)
        self.m = len(self.inputs)

        self.state = np.zeros(self.n)

        # state equation
        self.f = StateSpaceModel.F(self)

        self.lambdaA, self.lambdaB = StateSpaceModel.LinearizeInit(self)
        self.A = np.zeros([self.n, self.n])
        self.B = np.zeros([self.n, self.m])
        self.C = np.identity(self.n)
        self.D = np.zeros([self.n, self.m])
        StateSpaceModel.Linearize(self)

        self.ct_statespace = sp.signal.StateSpace(self.A, self.B, self.C, self.D)
        self.dt = 0
        self.dt_statespace


    def NonlinearDiscreteTimeStep(self, input_n):
        f_xu = StateSpaceModel.EvaluateNonlinear(self.state, input_n)
        self.state += f_xu*self.dt

    def LinearDiscreteTimeStep(self, input_n):
        self.state = self.phi@self.state + self.gamma@input_n

    def EvaluateNonlinear(self, state_numerical, input_numerical):
        lambda_f = sm.utilities.lambdify([self.statesym, self.inputs], self.f)
        f_evaluated = lambda_f(state_numerical, input_numerical)
        return f_evaluated

    def EvaluateNonLinearOnlyX(self, state_numerical):
        psi = 0
        theta = 0
        return psi, theta

    def F(self):
        f1 = sm.Matrix([self.statesym[3], self.statesym[4], self.statesym[5]])
        f2 = sm.Matrix([0, 0, -9.81])
        attachment_points = config.attachment_points
        for n, cable_tension in enumerate(self.inputs):
            f2 += StateSpaceModel.Force(self, cable_tension, attachment_points[n][0:3])

        f = sm.Matrix([f1,f2])
        return f

    def Linearize(self, x_eq=np.zeros(6), u_eq=np.zeros(8)):
        self.A = self.lambdaA(x_eq, u_eq)
        self.B = self.lambdaB(x_eq, u_eq)


    def LinearizeInit(self):
        A_ = sm.zeros(self.n)
        B_ = sm.zeros(self.n, self.m)

        for i, state_i in enumerate(self.statesym):
            A_[:, i] = sm.diff(self.f, state_i)

        for i, input_i in enumerate(self.inputs):
            B_[:, i] = sm.diff(self.f, input_i)

        lambda_A_ = sm.utilities.lambdify([self.statesym, self.inputs], A_)
        lambda_B_ = sm.utilities.lambdify([self.statesym, self.inputs], B_)

        return lambda_A_, lambda_B_

    def Force(self, cable_tension, cable_attachment_point):
        location = sm.Matrix([self.statesym[0], self.statesym[1], self.statesym[2]])
        center_of_mass_to_attachment_point = sm.Matrix([cable_attachment_point]).T - location
        unit_vector = center_of_mass_to_attachment_point/(sm.sqrt(center_of_mass_to_attachment_point.dot(center_of_mass_to_attachment_point)))
        force = unit_vector*cable_tension
        return force

    def CalculatePhiGamma(self):
        self.Phi = 0
        self.Gamma = 0

    def Setdt(self, dt):
        self.dt = dt
        StateSpaceModel.CalculatePhiGamma(self)

    def ResetState(self, state):
        self.state = state
