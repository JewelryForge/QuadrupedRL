import numpy as np
from matplotlib import pyplot as plt


def test_reward_reshape():
    x = np.arange(-2, 2, 0.01)
    plt.plot(x, np.tanh(x))
    plt.show()
