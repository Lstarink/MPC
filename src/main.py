import numpy as np
import scipy as sp
import math as mt
import nonlinear_state_space_model
import LQR_controller
import visualize

# dt = 0.01
# t = np.arange(0.0, 1, dt)
#
# model = nonlinear_state_space_model.StateSpaceModel()
# print(model.f[3])
# print(model.f[4])
# print(model.f[5])
# f_eq = 3*9.81 / (4 * mt.sqrt(3))
# x_eq = np.zeros(model.n)
# u_eq = np.array([f_eq, 0, f_eq, 0, f_eq, 0, f_eq, 0])
# model.Linearize(x_eq=x_eq, u_eq=u_eq)
#
# Q = np.identity(model.n)
# R = np.identity(model.m)
# controller = LQR_controller.LQR(model.Phi, model.Gamma, Q, R)
#
# ydt = np.zeros([model.n, len(t)])
# ynl = np.zeros([model.n, len(t)])
#
# x0 = np.ones(model.n)*0.5
#
# xplus_dt = x0
# xplus_nl = x0
#
# ydt[:, 0] = x0
# ynl[:, 0] = x0
# for n, tn in enumerate(t):
#     undt = controller.Tick(xplus_dt)
#     unnl = controller.Tick(xplus_nl)
#     xplus_dt = model.dt_tick(undt, xplus_dt)
#     xplus_nl = model.nl_tick(unnl, xplus_nl)
#     if n < len(t)-1:
#         ydt[:, n+1] = xplus_dt
#         ynl[:, n+1] = xplus_nl
#
#
# # visualize.VisualizeStateProgression(ydt, t)
# # visualize.VisualizeTrajectory3D(ydt[0, :], ydt[1, :], ydt[2, :])
# # visualize.VisualizeTrajectory3D(ynl[0, :], ynl[1, :], ynl[2, :])
# #
# visualize.VisualizeStateProgressionMultipleSims([ydt, ynl], t)



def LQR_sim():
    return 0

def MPC_sim():
    return 0