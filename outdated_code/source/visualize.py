import matplotlib.pyplot as plt
import numpy as np
import math as mt
import config_
import kinematics

def Visualize(location, orientation, ax):
    rib01 = np.array([config.c01B, config.c02B])
    rib02 = np.array([config.c01B, config.c03B])
    rib03 = np.array([config.c03B, config.c04B])
    rib04 = np.array([config.c02B, config.c04B])
    rib05 = np.array([config.c01B, config.c05B])
    rib06 = np.array([config.c02B, config.c06B])
    rib07 = np.array([config.c03B, config.c07B])
    rib08 = np.array([config.c04B, config.c08B])
    rib09 = np.array([config.c05B, config.c06B])
    rib10 = np.array([config.c05B, config.c07B])
    rib11 = np.array([config.c07B, config.c08B])
    rib12 = np.array([config.c06B, config.c08B])

    ribs = np.array([rib01, rib02, rib03, rib04,
                     rib05, rib06, rib07, rib08,
                     rib09, rib10, rib11, rib12])

    ribs = kinematics.RotateRibs(ribs, orientation)
    ribs = kinematics.TranslateRibs(ribs, location)

    for rib in ribs:
        x = np.array([rib[0][0], rib[1][0]])
        y = np.array([rib[0][1], rib[1][1]])
        z = np.array([rib[0][2], rib[1][2]])

        ax.plot(x, y, z, label='parametric curve', color= 'b')

    return 0


if __name__ == "__main__":
    ax = plt.figure().add_subplot(projection='3d')

    orientation = np.array([mt.pi/3,mt.pi/3,mt.pi/3])

    t = np.linspace(0,mt.pi/2, 10)
    x = np.cos(t)*10
    y = np.sin(t)*10
    z = 5*t

    phi = t*2
    psi = t*3
    theta = t

    for n, t_ in enumerate(t):
        location = np.array([x[n], y[n], z[n]])
        orientation = np.array([phi[n], psi[n], theta[n]])
        Visualize(location, orientation, ax)
    plt.show()
