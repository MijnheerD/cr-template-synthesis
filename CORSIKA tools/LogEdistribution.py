import numpy as np
import matplotlib.pyplot as plt


MIN_EXP = 17  # eV
MAX_EXP = 19  # eV
PLOT = False  # whether to plot distribution


def loguniform(minimum=17, maximum=19, size=(100,)):
    uniform_exp = np.random.uniform(minimum, maximum, size=size)
    return np.power(10, uniform_exp)


if PLOT:
    uniform_E = loguniform(MIN_EXP, MAX_EXP, size=(1000,))

    fig, ax = plt.subplots(1, 1)

    bins = np.power(10, np.linspace(MIN_EXP, MAX_EXP, 50))
    ax.hist(uniform_E, bins=bins)

    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Counts')
    ax.set_xscale('log')
    ax.set_title('Uniform log E distribution')

    plt.show()
