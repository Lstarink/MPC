import numpy as np
import scipy as sp
import math as mt
import nonlinear_state_space_model
import LQR_controller
import visualize
import saturate
import Observer

dt = 0.01
t = np.arange(0.0, 0.8, dt)

model = nonlinear_state_space_model.StateSpaceModel(dt=dt)

f_eq = 3*9.81 / (4 * mt.sqrt(3))
x_eq = np.zeros(model.n)
u_eq = np.array([f_eq, 0, f_eq, 0, f_eq, 0, f_eq, 0])
model.Linearize(x_eq=x_eq, u_eq=u_eq)

q1 = 5000
q2 = 10
Q = np.array([[q1, 0, 0, 0, 0, 0],
              [0, q1, 0, 0, 0, 0],
              [0, 0, q1, 0, 0, 0],
              [0, 0, 0, q2, 0, 0],
              [0, 0, 0, 0, q2, 0],
              [0, 0, 0, 0, 0, q2]])
R = np.identity(model.m)
controller = LQR_controller.LQR(model.Phi, model.Gamma, Q, R)

ydt = np.zeros([model.n, len(t)])
ynl = np.zeros([model.n, len(t)])

x0 = np.ones(model.n)*0.5

xplus_dt = x0
xplus_nl = x0

ydt[:, 0] = x0
ynl[:, 0] = x0
u_vector = np.zeros([8, len(t)])

poles = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.17, 0.18, 0.19, 0.15, 0.16, 0.14])*5

disturbance = np.array([0.00, 0.1, 0.00, 0.0, 0.0, 0.0])

estimated_state = x0
estimated_disturbance = disturbance

observer = Observer.Observer(dt, u_eq, x_eq, poles)
observer.InitState(x0, estimated_disturbance)
model.ResetState(x0)

estimated_state_vector = np.zeros([6, len(t)])
estimation_error= np.zeros([6, len(t)])
estimated_disturbance_vector= np.zeros([6, len(t)])
true_disturbance= np.zeros([6, len(t)])

for n, tn in enumerate(t):
    unnl = controller.Tick(estimated_state)
    unnl = saturate.Saturate(unnl, -u_eq, 100)
    u_vector[:, n] = unnl+u_eq
    xplus_nl = model.nl_tick(unnl, xplus_nl)
    estimated_state, estimated_disturbance = observer.Tick(xplus_nl+disturbance, unnl-u_eq)
    estimated_state = estimated_state-estimated_disturbance
    estimated_state_vector[:, n] = estimated_state
    estimation_error[:, n] = xplus_nl-estimated_state
    estimated_disturbance_vector[:, n] = estimated_disturbance
    true_disturbance[:, n] = disturbance
    if n < len(t)-1:
        ynl[:, n+1] = xplus_nl


# visualize.VisualizeStateProgression(ydt, t)
# visualize.VisualizeTrajectory3D(ydt[0, :], ydt[1, :], ydt[2, :])
# visualize.VisualizeTrajectory3D(ynl[0, :], ynl[1, :], ynl[2, :])
#
visualize.VisualizeStateProgressionMultipleSims([estimated_state_vector, ynl], t, handles=["State", "Estimated state"])
visualize.VisualizeStateProgressionMultipleSims([true_disturbance, estimated_disturbance_vector], t, lim=0.5, handles=["True disturbance", "Estimated disturbance"])
visualize.VisualizeInputs(u_vector, t)

#
# def LQR_sim():
#     return 0
#
# def MPC_sim():
#     return 0