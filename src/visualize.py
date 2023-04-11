import numpy as np
import matplotlib.pyplot as plt


def VisualizeTrajectory3D(x, y ,z ):
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(x, y, z, label='parametric curve')
    ax.legend()
    plt.show()

def VisualizeStateProgression(states, t, Title):
    fig, axs = plt.subplots(6,1)
    fig.suptitle(Title)
    ax_counter = 0
    for ax, state in zip(axs, states):
        ax.step(t, state)
        if ax_counter < 3:
            ax.set_ylim(bottom=-1.5, top=1.5)
        else:
            ax.set_ylim(bottom=-5, top=5)
        ax.grid()
        ax.set_xlabel("time [s]")
        ax.set_ylabel("state " + str(ax_counter + 1))
        ax_counter+=1
    plt.show()

def VisualizeStateProgressionMultipleSims(Sims, t, handles=None):
    fig, axs = plt.subplots(6,1)
    fig.suptitle('Vertically stacked subplots')
    labels = []
    for states in Sims:
        for n, state in enumerate(states):
            axs[n].step(t, state, label=str(n))
            labels.append(str(n))

    ax_counter = 0
    for ax in axs:
        ax.grid()
        ax.set_xlabel("time [s]")
        ax.set_ylabel("state " + str(ax_counter + 1))
        ax_counter += 1
    if handles:
        fig.legend(handles, labels, loc='upper center')
    plt.show()

def VisualizeInputs(U, t):
    fig, axs = plt.subplots(8,1)
    fig.suptitle('Inputs')
    ax_counter = 0
    for ax, input_n in zip(axs, U):
        ax.step(t, input_n)
        ax.grid()
        ax.set_xlabel("time [s]")
        ax.set_ylabel("u" + str(ax_counter + 1))
        ax_counter+=1
    plt.show()