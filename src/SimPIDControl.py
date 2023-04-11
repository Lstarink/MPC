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
import Observer

kp = 50.0
ki = 0.0
kd = 10
dt = 0.01
pretension = 0.0

statespace = nonlinear_state_space_model.StateSpaceModel(dt=dt)
inverse_kinematics = InverseKinematics.InverseKinematics(kp, kd, ki, dt, saturate=True, lower_limit=0.0)

state0 = np.ones(6)*0.5
statespace.ResetState(state0)

f_eq = 3*9.81 / (4 * mt.sqrt(3))
u_eq = np.array([f_eq, 0, f_eq, 0, f_eq, 0, f_eq, 0])
x_eq = np.zeros(6)
statespace.Linearize(u_eq=u_eq, x_eq=x_eq)

reference = np.array([0.0, 0.0, 0.0])
t = np.arange(0, .8, dt)

state = state0
x = np.zeros(t.shape)
y = np.zeros(t.shape)
z = np.zeros(t.shape)
state_vector = np.zeros([6, len(t)])
estimated_state_vector = np.zeros([6, len(t)])
input_vector = np.zeros([8, len(t)])
estimation_error = np.zeros([6, len(t)])
estimated_disturbance_vector = np.zeros([6, len(t)])
true_disturbance = np.zeros([6, len(t)])


ff_controller = feed_forward_controller.FeedForwardController(pretension)
fb_controller = pd_attempt2.PDXYZ(kp, ki, kd, dt)

poles = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.17, 0.18, 0.19, 0.15, 0.16, 0.14])*5

disturbance = np.array([0.00, 0.1, 0.00, 0.0, 0.0, 0.0])

estimated_state = state0
estimated_disturbance = disturbance

observer = Observer.Observer(dt, u_eq, x_eq, poles)
observer.InitState(state0, estimated_disturbance)
statespace.ResetState(state0)


for n in tqdm.tqdm(range(len(t))):
    tau_fb = fb_controller.Tick(reference, estimated_state)
    tau_ff = ff_controller.TickQuadProg(estimated_state[0:3])
    tau = tau_ff + tau_fb - u_eq
    input_vector[:, n] = tau + u_eq
    state = statespace.nl_tick(tau, state)
    estimated_state, estimated_disturbance = observer.Tick(state+disturbance, tau_fb)
    estimated_state = estimated_state-estimated_disturbance
    estimated_state_vector[:, n] = estimated_state
    state_vector[:, n] = state
    estimation_error[:, n] = state-estimated_state
    estimated_disturbance_vector[:, n] = estimated_disturbance
    true_disturbance[:, n] = disturbance
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

# visualize.VisualizeInputs(error_vector, t)
# visualize.VisualizeStateProgression(state_vector, t, "State progression")
# visualize.VisualizeStateProgression(estimated_state_vector, t, "Estimated state progression")
# visualize.VisualizeStateProgression(estimation_error, t, "Estimation error")
# visualize.VisualizeStateProgression(estimated_disturbance_vector, t, "Estimated disturbance")

# visualize.VisualizeStateProgression(state_vector_lin, t)

visualize.VisualizeInputs(input_vector, t)

visualize.VisualizeStateProgressionMultipleSims([state_vector, estimated_state_vector], t, handles=["State", "Estimated state"])
visualize.VisualizeStateProgressionMultipleSims([true_disturbance, estimated_disturbance_vector], t, lim=0.5, handles=["True disturbance", "Estimated disturbance"])
