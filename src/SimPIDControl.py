import numpy as np
import math as mt
import nonlinear_state_space_model
import InverseKinematics
import pd_attempt2
import config
import visualize
import feed_forward_controller
import tqdm
import saturate

kp = 100.0
ki = 3.0
kd = 20
dt = 0.05
pretension = 0.0

statespace = nonlinear_state_space_model.StateSpaceModel(dt=dt)
inverse_kinematics = InverseKinematics.InverseKinematics(kp, kd, ki, dt, saturate=True, lower_limit=0.0)

state0 = np.ones(6)*0.5
statespace.ResetState(state0)

f_eq = 3*9.81 / (4 * mt.sqrt(3))
u_eq = np.array([f_eq, 0, f_eq, 0, f_eq, 0, f_eq, 0])
statespace.Linearize(u_eq=u_eq)



reference = np.array([0.0, 0.0, 0.0])
t = np.arange(0, 1.0, dt)

state = state0
x = np.zeros(t.shape)
y = np.zeros(t.shape)
z = np.zeros(t.shape)
state_vector = np.zeros([6, len(t)])
input_vector = np.zeros([8, len(t)])
error_vector = np.zeros([8, len(t)])

ff_controller = feed_forward_controller.FeedForwardController(pretension)
fb_controller = pd_attempt2.PDXYZ(kp, ki, kd, dt)

for n, tn in enumerate(t):
    print(tn)
    # tau_fb, error = inverse_kinematics.Tick(reference, state[0:3])
    tau_fb = fb_controller.Tick(reference, state)
    # print(error.shape)
    tau_ff = ff_controller.TickQuadProg(state[0:3])
    tau = tau_ff + tau_fb - u_eq
    # error_vector[:, n] = error
    input_vector[:, n] = tau + u_eq
    state = statespace.nl_tick(tau, state)
    state_vector[:, n] = state
    x[n] = state[0]
    y[n] = state[1]
    z[n] = state[2]

state = state0
x_lin = np.zeros(t.shape)
y_lin = np.zeros(t.shape)
z_lin = np.zeros(t.shape)
state_vector_lin = np.zeros([6, len(t)])
input_vector_lin = np.zeros([8, len(t)])
error_vector_lin = np.zeros([8, len(t)])

# for n, tn in enumerate(t):
#     tau_fb, error = inverse_kinematics.Tick(reference, state[0:3])
#     tau_ff = ff_controller.Tick(reference)
#     tau = tau_ff + tau_fb - u_eq
#     error_vector_lin[:, n] = error
#     input_vector_lin[:, n] = tau
#     state = statespace.nl_tick(tau, state)
#     state_vector_lin[:, n] = state
#     x_lin[n] = state[0]
#     y_lin[n] = state[1]
#     z_lin[n] = state[2]
print(error_vector.shape)
print(input_vector.shape)
# visualize.VisualizeInputs(error_vector, t)
visualize.VisualizeStateProgression(state_vector, t)
# visualize.VisualizeStateProgression(state_vector_lin, t)

visualize.VisualizeInputs(input_vector, t)
