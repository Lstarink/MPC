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

def VisualizeStateProgressionMultipleSims(Sims, t, lim=None, handles=None):
    fig, axs = plt.subplots(6,1)
    labels = []

    for states in Sims:
        for n, state in enumerate(states):
            l,= axs[n].step(t, state, label=str(n))
            if n == 1:
                labels.append(l)

    ax_counter = 0
    for ax in axs:
        if lim:
            ax.set_ylim(bottom=-lim, top=lim)
        ax.grid()
        ax.set_xlabel("time [s]")
        ax.set_ylabel("state " + str(ax_counter + 1))
        ax_counter += 1
        if (ax_counter == 1) and handles:
            ax.legend(labels, handles, loc='upper right')
    plt.show()

def VisualizeInputs(U, t):
    fig, axs = plt.subplots(8,1)
    ax_counter = 0
    for ax, input_n in zip(axs, U):
        ax.step(t, input_n)
        ax.grid()
        ax.set_xlabel("time [s]")
        ax.set_ylabel("u" + str(ax_counter + 1))
        ax_counter+=1
    plt.show()

def VisualizeInputsMultipleSims(Sims, t, lim=None, handles=None):
    fig, axs = plt.subplots(4,2, figsize=(10,20))
    labels = []

    for states in Sims:
        for n, state in enumerate(states):
            if n < 4:
                l,= axs[n,0].step(t, state, label=str(n))
                if n == 1:
                    labels.append(l)
            else:
                l,= axs[n-4,1].step(t, state, label=str(n))
                if n == 1:
                    labels.append(l)

    ax_counter = 0
    for i in range(4):
        for j in range(2):
            if lim:
                axs[i,j].set_ylim(bottom=-lim, top=lim)
            axs[i,j].grid()
            axs[i,j].set_xlabel("time [s]")
            axs[i,j].set_ylabel("u " + str(ax_counter + 1))
            ax_counter += 1
            if (ax_counter == 2) and handles:
                axs[i,j].legend(labels, handles, loc='upper right')
    plt.show()