import numpy as np
import cvxpy as cp
import nonlinear_state_space_model


def otc_C_is_eye(y_ref, A, B, input_ub):
    # this function is expected to be used after lineariztion and discretization. Since the output matrix is an identity
    # matrix, the x_ref will be required directly from y_red.
    # y_ref: np.array 6 assume C = Identity
    # A: matrix 6*6  from linear discretized SS
    # B: array 6*8   from linear discretized SS
    # input_ub: int the upper bound of the input
    # return u_ref is an array 1*8 array

    # Attention: u_ref does not contain information from the linearization point u_eq.

    nx = A.shape[1]  # dimension of the state, 6
    nu = B.shape[1]  # dimension of the input, 8
    x_ref = y_ref
    P = np.eye(nu)  # weight for the cost function
    A_for_QP = B  # A_for_QP is not the state matrix, it is for the standard form of equality constraint AX = B
    B_for_QP = np.matmul((np.eye(nx) - A), x_ref)  # ditto
    G = np.concatenate((input_ub * np.eye(nu), -1 * np.eye(nu)), axis=0)
    h = np.concatenate((input_ub * np.ones((nu, 1)), -0.0001 * np.ones((nu, 1))), axis=0)
    h = h.reshape((2 * nu,))
    # that the lower bound is set to be 0.0001 is to guarantee the optimal values are all larger than zero; otherwise
    # (only) tiny negative optimal result would occur. Although essentially both of them would do, it is better to have
    # all of them positive.

    u = cp.Variable(nu)
    prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(u, P)),
                      [G @ u <= h,
                       A_for_QP @ u == B_for_QP])  # quadratic programming
    prob.solve()

    return u.value


def MakeTrajectory1():
    dt = 0.1
    t = np.arange(0, 10, dt)

    x = np.sin(t*0.5)
    y = np.cos(t*0.5)
    z = -0.5*np.ones(len(t))
    x_dot = 0.5*np.cos(t*0.5)
    y_dot = -0.5*np.sin(t*0.5)
    z_dot = np.zeros(len(t))


    for n, tn in enumerate(t):
        y_ref = np.array([x[n], y[n], z[n], x_dot[n], y_dot[n], z_dot[n]])



# # exmple 1: we want to steer the mass point to the origin.
# dt = 0.1
# statespace = nonlinear_state_space_model.StateSpaceModel()
# one_fourth_force = 9.81 * 0.25 * np.sqrt(3)
# statespace.Linearize(np.zeros(6),
#                      np.array([one_fourth_force, 0, one_fourth_force, 0, one_fourth_force, 0, one_fourth_force, 0]))
# statespace.Setdt(dt)
#
# A = statespace.Phi
# B = statespace.Gamma
#
# y_ref = np.array([0, 0, 0, 0, 0, 0])
# # Note that the latter three states should always be zero when we assign value for y_ref,
# # because y_ref should come from states at equilibrium, which implies the latter three velocity
# # state variable can not physically be zero.
# input_ub = 10
#
# uref = otc_C_is_eye(y_ref, A, B, input_ub)
#
# print(uref)
# # it prints [1.e-04 1.e-04 1.e-04 1.e-04 1.e-04 1.e-04 1.e-04 1.e-04], which stems from the lower bond
# # 0.0001, and could be considered as 0 or pre-tension. If we superimpose u_ref onto the u_eq, the sum
# # is the actual input reference for steering the state to the origin.
#
# # example 2: say we still linearize it at origin and the corresponding u_eq. But now, we want to steer it
# # to (0.2, 0, 0). That is to say, now y_ref = np.array([0.2, 0, 0, 0, 0, 0]).
#
# y_ref = np.array([0.2, 0, 0, 0, 0, 0])
# uref = otc_C_is_eye(y_ref, A, B, input_ub)
# print(uref)

# it prints [5.66480614e-01 5.66480614e-01 5.66480614e-01 5.66480614e-01, 1.00000000e-04 1.00000000e-04
# 1.00000000e-04 1.00000000e-04]. Obviously, the first four cables which correspond to the four corners of
# the positive x-axis are making effort. If we superimpose this u_ref onto the u_eq, the sum
# is the actual input reference for steering the state to the origin.
