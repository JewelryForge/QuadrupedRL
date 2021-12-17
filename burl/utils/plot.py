import time

import matplotlib.pyplot as plt
import numpy as np


def plotTrajectories(axis_range=(-0.3, 0.3, -0.4, -0.1)):
    plt.ion()
    fig, ax = plt.subplots(2, 2)
    axes = [*ax[0], *ax[1]]
    for ax in axes:
        ax.axis(axis_range)

    def plotOnce(dots_r, dots_b):
        for ax, dot_r, dot_b in zip(axes, dots_r, dots_b):
            ax.scatter(*dot_r, 10, c='r')
            ax.scatter(*dot_b, 10, c='b')
            plt.pause(1e-10)

    return plotOnce


plotter = plotTrajectories()

if __name__ == '__main__':
    for i in np.linspace(-1, 1, 100):
        plotter([(i, np.sin(i))], [(i, np.cos(i))])
        time.sleep(0.01)
