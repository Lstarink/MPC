import numpy as np
import nonlinear_state_space_model
import scipy as sp
import config
import cvxpy as cp

class FeedForwardController:
    def __init__(self, pretension):
        self.pretension = pretension

    def Tick(self, reference):
        upper_attachment_points = np.array([config.attachment_points[0],
                                            config.attachment_points[2],
                                            config.attachment_points[4],
                                            config.attachment_points[6]])

        lower_attachment_points = np.array([config.attachment_points[1],
                                            config.attachment_points[3],
                                            config.attachment_points[5],
                                            config.attachment_points[7]])

        unit_vectors_up = np.zeros([3, 4])
        unit_vectors_down = np.zeros([3, 4])

        for n, point in enumerate(upper_attachment_points):
            ref_to_point = point - reference
            unit_vectors_up[:, n] = ref_to_point/np.linalg.norm(ref_to_point)

        for n, point in enumerate(lower_attachment_points):
            ref_to_point = point - reference
            unit_vectors_down[:, n] = ref_to_point/np.linalg.norm(ref_to_point)

        upper = 1000.0
        bound = (self.pretension, None)

        A_eq = unit_vectors_up
        b_eq = np.array([0, 0, 9.81])-unit_vectors_down@(self.pretension*np.ones(4))
        c = np.ones(4)

        tau_ = sp.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=[bound, bound, bound, bound])
        tau = tau_.x
        for tau_n in tau:
            assert(tau_n >= 0)

        tau_ff = np.array([tau[0], self.pretension, tau[1], self.pretension, tau[2], self.pretension, tau[3], self.pretension])

        FeedForwardController.Test(self, reference, tau_ff)

        P = np.eye(4)
        u = cp.Variable(4)
        h = np.zeros(4)
        G = -1*np.identity(4)
        prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(u, P)),
                          [G @ u <= h,
                           A_eq @ u == b_eq])  # quadratic programming
        prob.solve()
        tau_quadprog = u.value
        tau_ff_quadprog = np.array([tau_quadprog[0], 0,
                                    tau_quadprog[1], 0,
                                    tau_quadprog[2], 0,
                                    tau_quadprog[3], 0])

        FeedForwardController.Test(self, reference, tau_ff_quadprog)

        return tau_ff

    def TickQuadProg(self, reference):
        upper_attachment_points = np.array([config.attachment_points[0],
                                            config.attachment_points[2],
                                            config.attachment_points[4],
                                            config.attachment_points[6]])

        lower_attachment_points = np.array([config.attachment_points[1],
                                            config.attachment_points[3],
                                            config.attachment_points[5],
                                            config.attachment_points[7]])

        unit_vectors_up = np.zeros([3, 4])
        unit_vectors_down = np.zeros([3, 4])

        for n, point in enumerate(upper_attachment_points):
            ref_to_point = point - reference
            unit_vectors_up[:, n] = ref_to_point/np.linalg.norm(ref_to_point)

        for n, point in enumerate(lower_attachment_points):
            ref_to_point = point - reference
            unit_vectors_down[:, n] = ref_to_point/np.linalg.norm(ref_to_point)

        upper = 1000.0
        bound = (self.pretension, None)

        A_eq = unit_vectors_up
        b_eq = np.array([0, 0, 9.81])-unit_vectors_down@(self.pretension*np.ones(4))

        P = np.eye(4)
        u = cp.Variable(4)
        h = np.zeros(4)
        G = -1*np.identity(4)
        prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(u, P)),
                          [G @ u <= h,
                           A_eq @ u == b_eq])  # quadratic programming
        prob.solve()
        tau_quadprog = u.value
        tau_ff_quadprog = np.array([tau_quadprog[0], 0,
                                    tau_quadprog[1], 0,
                                    tau_quadprog[2], 0,
                                    tau_quadprog[3], 0])

        FeedForwardController.Test(self, reference, tau_ff_quadprog)

        return tau_ff_quadprog


    def Test(self, reference, tau_ff):
        model = nonlinear_state_space_model.StateSpaceModel()
        x = np.array([reference[0], reference[1], reference[2],
                                        0.0, 0.0, 0.0])
        xplus = model.nl_tick(tau_ff, x)
        floatingpointerror = 1e-5
        np.testing.assert_allclose(xplus, x, atol=floatingpointerror)