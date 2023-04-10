import do_mpc as mpclib
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import matplotlib as mpl
import nonlinear_state_space_model as nlss
import math as mt
import visualize


def InitMPC(horizon, dt):
    def tvp_fun(tnow):
        n_horizon = 20
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

    Q = np.identity(6)
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
    mpc.bounds['upper', '_u', 'input_vector'] = upperu*np.ones(8) - u_eq

    mpc.setup()
    return mpc, model

n_horizon = 20
dt = 0.1

mpc, model = InitMPC(n_horizon, dt)

def tvp_fun_sim(tnow):
    tvp_template_sim['state_vector_ref'] = np.zeros(6)
    return tvp_template_sim

# Setup the simulator
simulator = mpclib.do_mpc.simulator.Simulator(model)
simulator.set_param(t_step = dt)
tvp_template_sim = simulator.get_tvp_template()
simulator.set_tvp_fun(tvp_fun_sim)
simulator.setup()

#Make a simulation
x0 = 0.9*np.array([1, 1, 1, 1, 1, 1]).reshape(-1, 1)
simulator.x0 = x0
mpc.x0 = x0
mpc.set_initial_guess()

mpl.rcParams['font.size'] = 18
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.grid'] = True

mpc_graphics = mpclib.do_mpc.graphics.Graphics(mpc.data)
sim_graphics = mpclib.do_mpc.graphics.Graphics(simulator.data)

fig, ax = plt.subplots(2, sharex=True, figsize=(16,9))
fig.align_ylabels()

for g in [sim_graphics, mpc_graphics]:
    g.add_line(var_type='_x', var_name='state_vector', axis=ax[0])
    g.add_line(var_type='_u', var_name='input_vector', axis=ax[1])

ax[0].set_ylabel('angle position [rad]')
ax[1].set_ylabel('motor angle [rad]')
ax[1].set_xlabel('time [s]')

sim_graphics.reset_axes()

# Reset and do the control loop
simulator.reset_history()
simulator.x0 = x0
mpc.reset_history()

f_eq = 3 * 9.81 / (4 * mt.sqrt(3))
u_eq = np.array([f_eq, 0, f_eq, 0, f_eq, 0, f_eq, 0])
nl_model = nlss.StateSpaceModel(dt=dt)
nl_model.ResetState(x0)
t = np.arange(0, 4, dt)
x_vec_lin = np.zeros([6,len(t)])

for i in range(len(t)):
    u0 = mpc.make_step(x0)
    x_vec_lin[:, i] = x0[:, 0]
    x0 = simulator.make_step(u0)

# mpc_graphics.plot_predictions(t_ind=0)
# sim_graphics.plot_results()
# sim_graphics.reset_axes()
# plt.show()
x0 = 0.9*np.array([1, 1, 1, 1, 1, 1]).reshape(-1, 1)
x_vec = np.zeros([6,len(t)])
nl_model.ResetState(x0)

for i in range(len(t)):
    u0 = mpc.make_step(x0)
    x_vec[:, i] = x0[:, 0]
    print(x0[:,0])
    x0 = nl_model.nl_tick(u0[:,0]+u_eq, x0[:,0]).reshape(-1, 1)

visualize.VisualizeStateProgressionMultipleSims([x_vec, x_vec_lin], t)