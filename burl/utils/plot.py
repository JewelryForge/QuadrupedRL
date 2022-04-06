import time

import numpy as np


def plot_trajectory(axis_range=(-0.3, 0.3, -0.4, -0.1)):
    import matplotlib.pyplot as plt
    plt.ion()
    fig, ax = plt.subplots(2, 2)
    axes = [*ax[0], *ax[1]]
    for ax in axes:
        ax.axis(axis_range)

    def plotOnce(subgraph_id, dot, color):
        axes[subgraph_id].scatter(*dot, 10, c=color)
        plt.pause(1e-10)

    return plotOnce


if __name__ == '__main__':
    plotter = plot_trajectory()
    for i in np.linspace(-1, 1, 100):
        plotter([(i, np.sin(i))], [(i, np.cos(i))])
        time.sleep(0.01)
