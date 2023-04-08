import numpy as np

import nonlinear_state_space_model
import InverseKinematics
import config
import visualize
import feed_forward_controller


kp = 0.0
ki = 0.0
kd = 0.00000
dt = 0.001
pretension = 1.0

statespace = nonlinear_state_space_model.StateSpaceModel(dt=dt)
inverse_kinematics = InverseKinematics.InverseKinematics(kp, kd, ki, dt, saturate=True, lower_limit=0.0)

state0 = np.array([0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0])
statespace.ResetState(state0)

reference = np.array([0.0, 0.5, 0.0])
t = np.arange(0, 5.0, dt)

state = state0
x = np.zeros(t.shape)
y = np.zeros(t.shape)
z = np.zeros(t.shape)
state_vector = np.zeros([6, len(t)])
input_vector = np.zeros([8, len(t)])
error_vector = np.zeros([8, len(t)])

ff_controler = feed_forward_controller.FeedForwardController(pretension)

for n, tn in enumerate(t):
    print("tn: ", tn)
    tau_fb, error = inverse_kinematics.Tick(reference, state[0:3])
    tau_ff = ff_controler.Tick(reference)
    tau = tau_ff + tau_fb
    error_vector[:, n] = error
    input_vector[:, n] = tau
    print("Tau: ", tau)
    state = statespace.nl_tick(tau, state)
    print("State: ", state)
    state_vector[:, n] = state
    x[n] = state[0]
    y[n] = state[1]
    z[n] = state[2]


visualize.VisualizeStateProgression(state_vector, t)
visualize.VisualizeInputs(input_vector, t)
visualize.VisualizeInputs(error_vector, t)