import numpy as np
import visualize

mpc50_x_lin = np.load("state_mpclin_50_true.npy")
mpc30_x_lin = np.load("state_mpclin_30_true.npy")
mpc50_x = np.load("state_mpc_n50_true.npy")
mpc30_x = np.load("state_mpc_n30_true.npy")
lqr_x = np.load("state_lqr_true.npy")
pd_x = np.load("state_pd_true.npy")

print(mpc30_x_lin.shape)
input("hold")
mpc50_u_lin = np.load("input_mpclin_50_true.npy")
mpc30_u_lin = np.load("input_mpclin_30_true.npy")
mpc50_u = np.load("input_mpc_n50_true.npy")
mpc30_u = np.load("input_mpc_n30_true.npy")
lqr_u = np.load("input_lqr_true.npy")
pd_u = np.load("input_pd_true.npy")

t_end = 2
dt = 0.01
n = int(t_end/dt)
t = np.arange(0, t_end, dt)
print(len(t))
print(mpc50_x[:,0:n].shape)
sims_x = [mpc50_x_lin[:,0:n], mpc30_x_lin[:,0:n], mpc50_x[:,0:n], mpc30_x[:,0:n]]

print(mpc50_x_lin[:,0:n].shape)
print(mpc30_x_lin.shape)
print(mpc50_x[:,0:n].shape)
print(mpc30_x[:,0:n].shape)
print(lqr_x[:,0:n].shape)
print(pd_x[:,0:n].shape)



input("hold")
for sim in sims_x:

    print(sim.shape)
sims_u = [mpc50_u_lin[:,0:n], mpc30_u_lin[:,0:n], mpc50_u[:,0:n], mpc30_u[:,0:n]]
handles = ["MPC Lin N=50", "MPC Lin N=30", "MPC NonLin N=50", "MPC NonLin N=30"]

visualize.VisualizeStateProgressionMultipleSims(sims_x, t, handles=handles)
visualize.VisualizeInputsMultipleSims(sims_u, t, handles=handles)

n = 80
sims_x = [mpc50_x_lin[:,0:n], mpc50_x[:,0:n], lqr_x[:,0:n]]
sims_u = [mpc50_u_lin[:,0:n], mpc50_u[:,0:n], lqr_u[:,0:n]]
handles = ["MPC Lin N=50", "MPC NonLin N=50", "LQR"]

visualize.VisualizeStateProgressionMultipleSims(sims_x, t[0:n], handles=handles)
visualize.VisualizeInputsMultipleSims(sims_u, t[0:n], handles=handles)