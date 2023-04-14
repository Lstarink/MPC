import numpy as np
import visualize

mpc50_x = np.load("state_mpc_n50.npy")
mpc30_x = np.load("state_mpc_n30.npy")
lqr_x = np.load("state_lqr.npy")
pd_x = np.load("state_pd.npy")

mpc50_u = np.load("input_mpc_n50.npy")
mpc30_u = np.load("input_mpc_n30.npy")
lqr_u = np.load("input_lqr.npy")
pd_u = np.load("input_pd.npy")

t_end = 2
dt = 0.01
n = int(t_end/dt)
t = np.arange(0, t_end, dt)
print(len(t))
print(mpc50_x[:,0:n].shape)
sims_x = [mpc50_x[:,0:n], mpc30_x[:,0:n], lqr_x[:,0:n], pd_x[:,0:n]]
sims_u = [mpc50_u[:,0:n], mpc30_u[:,0:n], lqr_u[:,0:n], pd_u[:,0:n]]
handles = ["MPC N=50", "MPC N=30", "LQR", "PD"]

visualize.VisualizeStateProgressionMultipleSims(sims_x, t, handles=handles)
visualize.VisualizeInputsMultipleSims(sims_u, t, handles=handles)