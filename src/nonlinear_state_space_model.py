import sympy as sm
import math as mt
import numpy as np
import config
import scipy as sp
import warnings
import control


class StateSpaceModel:
    def __init__(self, dt=0.1):

        self.dt = dt
        # states
        x, y, z, x_dot, y_dot, z_dot = sm.symbols("x, y, z, x_dot, y_dot, z_dot")
        self.statesym = sm.Matrix([x, y, z, x_dot, y_dot, z_dot])

        # inputs
        c1, c2, c3, c4, c5, c6, c7, c8 = sm.symbols("c1, c2, c3, c4, c5, c6, c7, c8")
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

        self.x_eq = np.zeros(self.n)
        self.u_eq = np.zeros(self.n)
        StateSpaceModel.Linearize(self)

        self.ct_statespace = sp.signal.StateSpace(self.A, self.B, self.C, self.D)
        self.dt_statespace = control.StateSpace(self.A, self.B, self.C, self.D, dt=self.dt)


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
        f_eq =  StateSpaceModel.EvaluateNonlinear(self, x_eq, u_eq)

        try:
            np.testing.assert_array_equal(np.zeros(self.n), f_eq[:,0])
        except AssertionError:
            warnings.warn('The point you are linearizing around is not an equilibrium\n')

        self.A = self.lambdaA(x_eq, u_eq)
        self.B = self.lambdaB(x_eq, u_eq)
        self.dt_statespace = control.StateSpace(self.A, self.B, self.C, self.D, dt=self.dt)
        self.ct_statespace = sp.signal.StateSpace(self.A, self.B, self.C, self.D)


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

    def ct_sim(self, u, t, x0):
        out = sp.signal.lsim(self.ct_statespace, u, t, X0=x0)
        return out[1]

    def dt_sim(self, u, t, x0):
        y = np.zeros([len(t), self.n])
        x_plus = x0
        for n, t_n in enumerate(t):
            x_plus = self.dt_statespace.dynamics(t_n, x_plus, u[n, :])
            y[n, :] = x_plus
        return y

    def Force(self, cable_tension, cable_attachment_point):
        location = sm.Matrix([self.statesym[0], self.statesym[1], self.statesym[2]])
        center_of_mass_to_attachment_point = sm.Matrix([cable_attachment_point]).T - location
        unit_vector = center_of_mass_to_attachment_point/(sm.sqrt(center_of_mass_to_attachment_point.dot(center_of_mass_to_attachment_point)))
        force = unit_vector*cable_tension
        return force

    def Setdt(self, dt):
        self.dt = dt
        [Phi, Gamma, C, D] = sp.signal.cont2discrete(self.ct_statespace, dt)
        print(Phi)
        print(Gamma)
        self.dt_statespace = sp.signal.StateSpace(Phi, Gamma, self.C, self.D, self.dt)

    def ResetState(self, state):
        self.state = state
