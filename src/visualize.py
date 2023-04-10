import numpy as np
import matplotlib.pyplot as plt


def VisualizeTrajectory3D(x, y ,z ):
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(x, y, z, label='parametric curve')
    ax.legend()
    plt.show()

def VisualizeStateProgression(states, t):
    fig, axs = plt.subplots(6,1)
    fig.suptitle('Vertically stacked subplots')
    for ax, state in zip(axs, states):
        ax.plot(t, state)
        ax.set_ylim(bottom=-1.0, top=1.0)
        ax.grid()
    plt.show()

def VisualizeStateProgressionMultipleSims(Sims, t, handles=None):
    fig, axs = plt.subplots(6,1)
    fig.suptitle('Vertically stacked subplots')
    labels = []
    for states in Sims:
        for n, state in enumerate(states):
            axs[n].step(t, state, label=str(n))
            labels.append(str(n))
    if handles:
        fig.legend(handles, labels, loc='upper center')
    plt.show()

def VisualizeInputs(U, t):
    fig, axs = plt.subplots(8,1)
    fig.suptitle('Inputs')
    for ax, input_n in zip(axs, U):
        ax.plot(t, input_n)
    plt.show()