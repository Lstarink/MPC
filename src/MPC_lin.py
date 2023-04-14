import do_mpc as mpclib
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import matplotlib as mpl
import nonlinear_state_space_model as nlss
import math as mt
import visualize
import tqdm
import Observer


def InitMPC(horizon, dt):
    def tvp_fun(tnow):
        n_horizon = 10
        for k in range(n_horizon + 1):
            tvp_template['_tvp', k, 'state_vector_ref'] = np.zeros(6)
        return tvp_template

    model_type = 'discrete'
    model = mpclib.do_mpc.model.Model(model_type)

    nonlinear_model = nlss.StateSpaceModel(dt=dt)
    x_eq = np.zeros(6)
    f_eq = 3 * 9.81 / (4 * mt.sqrt(3))
    u_eq = np.array([f_eq, 0, f_eq, 0, f_eq, 0, f_eq, 0])
    nonlinear_model.Linearize(x_eq=x_eq, u_eq=u_eq)
    Phi = nonlinear_model.Phi
    Gamma = nonlinear_model.Gamma

    # #Set optimal inputs
    state_vector_ref = model.set_variable(var_type='_tvp', var_name='state_vector_ref', shape=(6, 1))

    state_vector = model.set_variable(var_type='_x', var_name='state_vector', shape=(6, 1))
    input_vector = model.set_variable(var_type='_u', var_name='input_vector', shape=(8, 1))

    model.set_rhs('state_vector', Phi@state_vector + Gamma@input_vector)
    model.setup()

    mpc = mpclib.do_mpc.controller.MPC(model)

    # Setup Controller
    setup_mpc = {
        'n_horizon': horizon,
        't_step': dt,
        'store_full_solution': True,
    }
    mpc.set_param(**setup_mpc)

    q1 = 5000
    q2 = 10
    Q = np.array([[q1, 0, 0, 0, 0, 0],
                  [0, q1, 0, 0, 0, 0],
                  [0, 0, q1, 0, 0, 0],
                  [0, 0, 0, q2, 0, 0],
                  [0, 0, 0, 0, q2, 0],
                  [0, 0, 0, 0, 0, q2]])
    m_term = (state_vector).T @ Q @ (state_vector)
    l_term = (state_vector).T @ Q @ (state_vector)
    R = np.identity(8)
    mpc.set_objective(mterm=m_term, lterm=l_term)
    mpc.set_rterm(input_vector=np.ones(8))

    # define the reference
    tvp_template = mpc.get_tvp_template()

    mpc.set_tvp_fun(tvp_fun)

    lowerx = -1
    upperx = 1
    lowerdot = -10
    upperdot = 10
    loweru = 0
    upperu = 100

    # Set constraints
    mpc.bounds['lower', '_x', 'state_vector'] = np.array([lowerx, lowerx, lowerx, lowerdot, lowerdot, lowerdot])
    mpc.bounds['upper', '_x', 'state_vector'] = np.array([upperx, upperx, upperx, upperdot, upperdot, upperdot])
    mpc.bounds['lower', '_u', 'input_vector'] = loweru*np.ones(8) - u_eq
    mpc.bounds['upper', '_u', 'input_vector'] = upperu*np.ones(8)

    mpc.setup()
    return mpc, model

n_horizon = 10
dt = 0.01

mpc, model = InitMPC(n_horizon, dt)

statespace = nlss.StateSpaceModel(dt=dt)

state0 = np.ones(6)*0.5
statespace.ResetState(state0)

f_eq = 3*9.81 / (4 * mt.sqrt(3))
u_eq = np.array([f_eq, 0, f_eq, 0, f_eq, 0, f_eq, 0])
x_eq = np.zeros(6)
statespace.Linearize(u_eq=u_eq, x_eq=x_eq)

reference = np.array([0.0, 0.0, 0.0])
t = np.arange(0, 2.0, dt)

state = state0
state_vector = np.zeros([6, len(t)])
estimated_state_vector = np.zeros([6, len(t)])
input_vector = np.zeros([8, len(t)])
estimation_error = np.zeros([6, len(t)])
estimated_disturbance_vector = np.zeros([6, len(t)])
true_disturbance = np.zeros([6, len(t)])

poles = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.17, 0.14, 0.19, 0.15, 0.16, 0.14])*5

disturbance = np.array([0.00, 0.0, 0.00, 0.0, 0.0, 0.0])

estimated_state = state0
estimated_disturbance = disturbance

observer = Observer.Observer(dt, u_eq, x_eq, poles)
observer.InitState(state0, estimated_disturbance)
statespace.ResetState(state0)

mpc.x0 = state0
mpc.set_initial_guess()
for n in tqdm.tqdm(range(len(t))):
    tau = mpc.make_step(state)
    tau = tau[:, 0]
    input_vector[:, n] = tau+u_eq
    estimated_state, estimated_disturbance = observer.Tick(state+disturbance, tau)

    state = statespace.nl_tick(tau, state)
    # estimated_state = statespace.nl_tick(tau, estimated_state)
    estimated_state = estimated_state-estimated_disturbance
    estimated_state_vector[:, n] = estimated_state
    state_vector[:, n] = state
    estimation_error[:, n] = state-estimated_state
    estimated_disturbance_vector[:, n] = estimated_disturbance
    true_disturbance[:, n] = disturbance

np.save("state_mpclin_10_true.npy", state_vector)
np.save("input_mpclin_10_true.npy", input_vector)
visualize.VisualizeStateProgressionMultipleSims([state_vector, estimated_state_vector], t, handles=["State", "Estimated state"])
visualize.VisualizeStateProgressionMultipleSims([true_disturbance, estimated_disturbance_vector], t, lim=0.5, handles=["True disturbance", "Estimated disturbance"])
visualize.VisualizeInputs(input_vector, t)