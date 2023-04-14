import do_mpc as mpclib
import numpy as np
import casadi as ca
import tqdm
import nonlinear_state_space_model
import Observer
import visualize
import math as mt

def InitMPC(horizon, dt):
    def tvp_fun(tnow):
        n_horizon = 50

        for k in range(n_horizon + 1):
            tvp_template['_tvp', k, 'x_ref'] = 0#-np.cos(tnow * 0.5)*0.8
            tvp_template['_tvp', k, 'y_ref'] = 0#-np.sin(tnow * 0.5)*0.8
            tvp_template['_tvp', k, 'z_ref'] = 0#-0.5
            tvp_template['_tvp', k, 'x_dot_ref'] = 0#0.5*np.sin(tnow * 0.5)*0.8
            tvp_template['_tvp', k, 'y_dot_ref'] = 0#-0.5*np.cos(tnow * 0.5)*0.8
            tvp_template['_tvp', k, 'z_dot_ref'] = 0#0

            tvp_template['_tvp', k, 'c1_opt'] = 0
            tvp_template['_tvp', k, 'c1_opt'] = 0
            tvp_template['_tvp', k, 'c1_opt'] = 0
            tvp_template['_tvp', k, 'c1_opt'] = 0
            tvp_template['_tvp', k, 'c1_opt'] = 0
            tvp_template['_tvp', k, 'c1_opt'] = 0
            tvp_template['_tvp', k, 'c1_opt'] = 0
            tvp_template['_tvp', k, 'c1_opt'] = 0
        return tvp_template

    model_type = 'continuous'
    model = mpclib.do_mpc.model.Model(model_type)

    # Set the states
    x = model.set_variable(var_type='_x', var_name='x', shape=(1, 1))
    y = model.set_variable(var_type='_x', var_name='y', shape=(1, 1))
    z = model.set_variable(var_type='_x', var_name='z', shape=(1, 1))
    x_dot = model.set_variable(var_type='_x', var_name='x_dot', shape=(1, 1))
    y_dot = model.set_variable(var_type='_x', var_name='y_dot', shape=(1, 1))
    z_dot = model.set_variable(var_type='_x', var_name='z_dot', shape=(1, 1))

    # Set inputs
    c1 = model.set_variable(var_type='_u', var_name='c1', shape=(1, 1))
    c2 = model.set_variable(var_type='_u', var_name='c2', shape=(1, 1))
    c3 = model.set_variable(var_type='_u', var_name='c3', shape=(1, 1))
    c4 = model.set_variable(var_type='_u', var_name='c4', shape=(1, 1))
    c5 = model.set_variable(var_type='_u', var_name='c5', shape=(1, 1))
    c6 = model.set_variable(var_type='_u', var_name='c6', shape=(1, 1))
    c7 = model.set_variable(var_type='_u', var_name='c7', shape=(1, 1))
    c8 = model.set_variable(var_type='_u', var_name='c8', shape=(1, 1))

    # Set references
    x_ref = model.set_variable(var_type='_tvp', var_name='x_ref')
    y_ref = model.set_variable(var_type='_tvp', var_name='y_ref')
    z_ref = model.set_variable(var_type='_tvp', var_name='z_ref')
    x_dot_ref = model.set_variable(var_type='_tvp', var_name='x_dot_ref')
    y_dot_ref = model.set_variable(var_type='_tvp', var_name='y_dot_ref')
    z_dot_ref = model.set_variable(var_type='_tvp', var_name='z_dot_ref')

    #Set optimal inputs
    c1_opt = model.set_variable(var_type='_tvp', var_name='c1_opt')
    c2_opt = model.set_variable(var_type='_tvp', var_name='c2_opt')
    c3_opt = model.set_variable(var_type='_tvp', var_name='c3_opt')
    c4_opt = model.set_variable(var_type='_tvp', var_name='c4_opt')
    c5_opt = model.set_variable(var_type='_tvp', var_name='c5_opt')
    c6_opt = model.set_variable(var_type='_tvp', var_name='c6_opt')
    c7_opt = model.set_variable(var_type='_tvp', var_name='c7_opt')
    c8_opt = model.set_variable(var_type='_tvp', var_name='c8_opt')

    model.set_rhs('x', x_dot)
    model.set_rhs('y', y_dot)
    model.set_rhs('z', z_dot)
    model.set_rhs('x_dot', c1 * (1 - x) / ca.sqrt((1 - x) ** 2 + (1 - y) ** 2 + (1 - z) ** 2) + c2 * (1 - x) / ca.sqrt(
        (1 - x) ** 2 + (1 - y) ** 2 + (-z - 1) ** 2) + c3 * (1 - x) / ca.sqrt(
        (1 - x) ** 2 + (1 - z) ** 2 + (-y - 1) ** 2) + c4 * (1 - x) / ca.sqrt(
        (1 - x) ** 2 + (-y - 1) ** 2 + (-z - 1) ** 2) + c5 * (-x - 1) / ca.sqrt(
        (1 - y) ** 2 + (1 - z) ** 2 + (-x - 1) ** 2) + c6 * (-x - 1) / ca.sqrt(
        (1 - y) ** 2 + (-x - 1) ** 2 + (-z - 1) ** 2) + c7 * (-x - 1) / ca.sqrt(
        (1 - z) ** 2 + (-x - 1) ** 2 + (-y - 1) ** 2) + c8 * (-x - 1) / ca.sqrt(
        (-x - 1) ** 2 + (-y - 1) ** 2 + (-z - 1) ** 2))
    model.set_rhs('y_dot', c1 * (1 - y) / ca.sqrt((1 - x) ** 2 + (1 - y) ** 2 + (1 - z) ** 2) + c2 * (1 - y) / ca.sqrt(
        (1 - x) ** 2 + (1 - y) ** 2 + (-z - 1) ** 2) + c3 * (-y - 1) / ca.sqrt(
        (1 - x) ** 2 + (1 - z) ** 2 + (-y - 1) ** 2) + c4 * (-y - 1) / ca.sqrt(
        (1 - x) ** 2 + (-y - 1) ** 2 + (-z - 1) ** 2) + c5 * (1 - y) / ca.sqrt(
        (1 - y) ** 2 + (1 - z) ** 2 + (-x - 1) ** 2) + c6 * (1 - y) / ca.sqrt(
        (1 - y) ** 2 + (-x - 1) ** 2 + (-z - 1) ** 2) + c7 * (-y - 1) / ca.sqrt(
        (1 - z) ** 2 + (-x - 1) ** 2 + (-y - 1) ** 2) + c8 * (-y - 1) / ca.sqrt(
        (-x - 1) ** 2 + (-y - 1) ** 2 + (-z - 1) ** 2))
    model.set_rhs('z_dot', c1 * (1 - z) / ca.sqrt((1 - x) ** 2 + (1 - y) ** 2 + (1 - z) ** 2) + c2 * (-z - 1) / ca.sqrt(
        (1 - x) ** 2 + (1 - y) ** 2 + (-z - 1) ** 2) + c3 * (1 - z) / ca.sqrt(
        (1 - x) ** 2 + (1 - z) ** 2 + (-y - 1) ** 2) + c4 * (-z - 1) / ca.sqrt(
        (1 - x) ** 2 + (-y - 1) ** 2 + (-z - 1) ** 2) + c5 * (1 - z) / ca.sqrt(
        (1 - y) ** 2 + (1 - z) ** 2 + (-x - 1) ** 2) + c6 * (-z - 1) / ca.sqrt(
        (1 - y) ** 2 + (-x - 1) ** 2 + (-z - 1) ** 2) + c7 * (1 - z) / ca.sqrt(
        (1 - z) ** 2 + (-x - 1) ** 2 + (-y - 1) ** 2) + c8 * (-z - 1) / ca.sqrt(
        (-x - 1) ** 2 + (-y - 1) ** 2 + (-z - 1) ** 2) - 9.81)

    model.setup()

    mpc = mpclib.do_mpc.controller.MPC(model)

    # Setup Controller
    setup_mpc = {
        'n_horizon': horizon,
        't_step': dt,
        'store_full_solution': True,
    }
    mpc.set_param(**setup_mpc)


    a = 1
    b = 0.0
    mterm = a*((x) ** 2 + (y) ** 2 + (z) ** 2) + b*((x_dot) ** 2 + (y_dot) ** 2 + (z_dot) ** 2)
    lterm = a*((x) ** 2 + (y) ** 2 + (z) ** 2) + b*((x_dot) ** 2 + (y_dot) ** 2 + (z_dot) ** 2)
    r = 0.1
    mpc.set_objective(mterm=mterm, lterm=0.5*lterm)
    mpc.set_rterm(c1=r, c2=r, c3=r, c4=r, c5=r, c6=r, c7=r, c8=r)

    # define the reference
    tvp_template = mpc.get_tvp_template()

    mpc.set_tvp_fun(tvp_fun)

    lowerx = -1
    upperx = 1
    lowerdot = -10
    upperdot = 10
    loweru = 0
    upperu = 10000

    # Set constraints
    # Lower bounds on states
    mpc.bounds['lower', '_x', 'x'] = lowerx
    mpc.bounds['lower', '_x', 'y'] = lowerx
    mpc.bounds['lower', '_x', 'z'] = lowerx
    mpc.bounds['lower', '_x', 'x_dot'] = lowerdot
    mpc.bounds['lower', '_x', 'y_dot'] = lowerdot
    mpc.bounds['lower', '_x', 'z_dot'] = lowerdot
    # Upper bounds on states
    mpc.bounds['upper', '_x', 'x'] = upperx
    mpc.bounds['upper', '_x', 'y'] = upperx
    mpc.bounds['upper', '_x', 'z'] = upperx
    mpc.bounds['upper', '_x', 'x_dot'] = upperdot
    mpc.bounds['upper', '_x', 'y_dot'] = upperdot
    mpc.bounds['upper', '_x', 'z_dot'] = upperdot

    # Lower bounds on inputs:
    mpc.bounds['lower', '_u', 'c1'] = loweru
    mpc.bounds['lower', '_u', 'c2'] = loweru
    mpc.bounds['lower', '_u', 'c3'] = loweru
    mpc.bounds['lower', '_u', 'c4'] = loweru
    mpc.bounds['lower', '_u', 'c5'] = loweru
    mpc.bounds['lower', '_u', 'c6'] = loweru
    mpc.bounds['lower', '_u', 'c7'] = loweru
    mpc.bounds['lower', '_u', 'c8'] = loweru
    # Lower bounds on inputs:
    mpc.bounds['upper', '_u', 'c1'] = upperu
    mpc.bounds['upper', '_u', 'c2'] = upperu
    mpc.bounds['upper', '_u', 'c3'] = upperu
    mpc.bounds['upper', '_u', 'c4'] = upperu
    mpc.bounds['upper', '_u', 'c5'] = upperu
    mpc.bounds['upper', '_u', 'c6'] = upperu
    mpc.bounds['upper', '_u', 'c7'] = upperu
    mpc.bounds['upper', '_u', 'c8'] = upperu

    mpc.setup()
    return mpc, model


n_horizon = 50
dt = 0.01

mpc, model = InitMPC(n_horizon, dt)
statespace = nonlinear_state_space_model.StateSpaceModel(dt=dt)

state0 = np.ones(6)*0.5
statespace.ResetState(state0)

f_eq = 3*9.81 / (4 * mt.sqrt(3))
u_eq = np.array([f_eq, 0, f_eq, 0, f_eq, 0, f_eq, 0])
x_eq = np.zeros(6)
statespace.Linearize(u_eq=u_eq, x_eq=x_eq)

reference = np.array([0.0, 0.0, 0.0])
t = np.arange(0, 5.0, dt)

state = state0
state_vector = np.zeros([6, len(t)])
estimated_state_vector = np.zeros([6, len(t)])
input_vector = np.zeros([8, len(t)])
estimation_error = np.zeros([6, len(t)])
estimated_disturbance_vector = np.zeros([6, len(t)])
true_disturbance = np.zeros([6, len(t)])

poles = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.17, 0.18, 0.19, 0.15, 0.16, 0.14])*5
# poles = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#                   0.95, 0.95, 0.95, 0.95, 0.95, 0.95])*0.2
disturbance = np.array([0.00, 0.1, 0.00, 0.0, 0.0, 0.0])

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
    input_vector[:, n] = tau
    state = statespace.nl_tick(tau-u_eq, state)
    estimated_state, estimated_disturbance = observer.Tick(state+disturbance, tau-u_eq)
    estimated_state = estimated_state-estimated_disturbance
    estimated_state_vector[:, n] = estimated_state
    state_vector[:, n] = state
    estimation_error[:, n] = state-estimated_state
    estimated_disturbance_vector[:, n] = estimated_disturbance
    true_disturbance[:, n] = disturbance

visualize.VisualizeStateProgressionMultipleSims([state_vector, estimated_state_vector], t, handles=["State", "Estimated state"])
visualize.VisualizeStateProgressionMultipleSims([true_disturbance, estimated_disturbance_vector], t, lim=0.5, handles=["True disturbance", "Estimated disturbance"])
visualize.VisualizeInputs(input_vector, t)

np.save("state_mpc_n50_true.npy", state_vector)
np.save("input_mpc_n50_true.npy", input_vector)

# def tvp_fun_sim(tnow):
#     tvp_template_sim['x_ref'] = -0.5
#     tvp_template_sim['y_ref'] = -0.5
#     tvp_template_sim['z_ref'] = -0.5
#     return tvp_template_sim
#
# # Setup the simulator
# simulator = mpclib.do_mpc.simulator.Simulator(model)
# simulator.set_param(t_step = dt)
# tvp_template_sim = simulator.get_tvp_template()
# simulator.set_tvp_fun(tvp_fun_sim)
# simulator.setup()
#
# #Make a simulation
# x0 = 0.5*np.array([1, 1, 1, 1, 1, 1]).reshape(-1, 1)
# simulator.x0 = x0
# mpc.x0 = x0
# mpc.set_initial_guess()
#
# mpl.rcParams['font.size'] = 18
# mpl.rcParams['lines.linewidth'] = 3
# mpl.rcParams['axes.grid'] = True
#
# mpc_graphics = mpclib.do_mpc.graphics.Graphics(mpc.data)
# sim_graphics = mpclib.do_mpc.graphics.Graphics(simulator.data)
#
# fig, ax = plt.subplots(2, sharex=True, figsize=(16,9))
# fig.align_ylabels()
#
# for g in [sim_graphics, mpc_graphics]:
#     # Plot the angle positions (phi_1, phi_2, phi_2) on the first axis:
#     g.add_line(var_type='_x', var_name='x', axis=ax[0])
#     g.add_line(var_type='_x', var_name='y', axis=ax[0])
#     g.add_line(var_type='_x', var_name='z', axis=ax[0])
#
#     # Plot the set motor positions (phi_m_1_set, phi_m_2_set) on the second axis:
#     g.add_line(var_type='_u', var_name='c1', axis=ax[1])
#     g.add_line(var_type='_u', var_name='c2', axis=ax[1])
#     g.add_line(var_type='_u', var_name='c3', axis=ax[1])
#     g.add_line(var_type='_u', var_name='c4', axis=ax[1])
#     g.add_line(var_type='_u', var_name='c5', axis=ax[1])
#     g.add_line(var_type='_u', var_name='c6', axis=ax[1])
#     g.add_line(var_type='_u', var_name='c7', axis=ax[1])
#     g.add_line(var_type='_u', var_name='c8', axis=ax[1])
#
# ax[0].set_ylabel('angle position [rad]')
# ax[1].set_ylabel('motor angle [rad]')
# ax[1].set_xlabel('time [s]')
#
# u0 = np.zeros((8,1))
# for i in range(200):
#     simulator.make_step(u0)
#
# sim_graphics.plot_results()
# # Reset the limits on all axes in graphic to show the data.
# sim_graphics.reset_axes()
# # Show the figure:
# # plt.show()
#
# u0 = mpc.make_step(x0)
# sim_graphics.clear()
#
# mpc_graphics.plot_predictions()
# mpc_graphics.reset_axes()
# # Show the figure:
# # plt.show()
#
# # Reset and do the control loop
# simulator.reset_history()
# simulator.x0 = x0
# mpc.reset_history()
#
# for i in range(20):
#     u0 = mpc.make_step(x0)
#     # disturbance = np.random.normal(0.0, 0.01, size=6)
#     x0 = simulator.make_step(u0)
#     # print(disturbance)
#
# # Plot predictions from t=0
# mpc_graphics.plot_predictions(t_ind=0)
# # Plot results until current time
# sim_graphics.plot_results()
# sim_graphics.reset_axes()
# plt.show()
#
#
