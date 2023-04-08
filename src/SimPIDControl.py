import numpy as np

import nonlinear_state_space_model
import InverseKinematics
import config
import visualize

kp = 0.1
ki = 0.0
kd = 0.0
dt = 0.1

statespace = nonlinear_state_space_model.StateSpaceModel()
inverse_kinematics = InverseKinematics.InverseKinematics(kp, kd, ki, dt, saturate=True, lower_limit=0.0)

state0 = np.zeros(6)
statespace.ResetState(state0)

reference = np.array([0.0, 0.0, 0.5])
t = np.arange(0, 10.0, dt)

state = state0
x = np.zeros(t.shape)
y = np.zeros(t.shape)
z = np.zeros(t.shape)
state_vector = np.zeros([6, len(t)])
input_vector = np.zeros([8, len(t)])
error_vector = np.zeros([8, len(t)])

for n, tn in enumerate(t):
    tau, error = inverse_kinematics.Tick(reference, state[0:3])
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