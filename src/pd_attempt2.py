import numpy as np
import pid_controller
import config
import cvxpy as cp

class PDXYZ:
    def __init__(self, kp, ki, kd, dt):
        self.controllers = np.array([pid_controller.PID_Controller(kp, ki, kd, dt),
                                     pid_controller.PID_Controller(kp, ki, kd, dt),
                                     pid_controller.PID_Controller(kp, ki, kd, dt)])
        self.kp = kp
        self.kd = kd

    def Tick(self, reference, state):
        error = reference-state[0:3]
        error_vel = -state[3:6]
        f_xyz = np.zeros(3)
        location = state[0:3]
        for n, e_n in enumerate(error):
            # f_xyz[n] = self.controllers[n].Tick(e_n)
            f_xyz[n] = self.kp*e_n + self.kd*error_vel[n]

        tau = PDXYZ.F_xyz_To_tau(self, f_xyz, location)
        return tau


    def F_xyz_To_tau(self, f_xyz, state):
        unit_vectors = np.zeros([3, 8])

        for n, point in enumerate(config.attachment_points):
            ref_to_point = point - state
            unit_vectors[:, n] = ref_to_point/np.linalg.norm(ref_to_point)


        A_eq = unit_vectors
        b_eq = f_xyz

        P = np.eye(8)
        u = cp.Variable(8)
        h = np.zeros(8)
        G = -1*np.identity(8)
        prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(u, P)),
                          [G @ u <= h,
                           A_eq @ u == b_eq])  # quadratic programming
        prob.solve()
        tau_quadprog = u.value
        return(tau_quadprog)

