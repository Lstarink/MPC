import numpy as np
import matplotlib.pyplot as plt


def VisualizeTrajectory3D(x, y ,z ):
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(x, y, z, label='parametric curve')
    ax.legend()
    plt.show()

def VisualizeStatePrograssion(states, t):
    fig, axs = plt.subplots(6,1)
    fig.suptitle('Vertically stacked subplots')
    for ax, state in zip(axs, states):
        ax.plot(t, state)
    plt.show()