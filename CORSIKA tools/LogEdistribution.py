import numpy as np
import matplotlib.pyplot as plt


MIN_EXP = 17  # eV
MAX_EXP = 19  # eV
PLOT = False  # whether to plot distribution


def loguniform(minimum=17, maximum=19, size=(100,), return_exp=False, base=10):
    """
    Draw exponents from uniform distribution over the interval [minimum, maximum). If return_exp is False (the default),
    the returned array will contain the results of raising the base using the sampled exponents. Otherwise, the
    exponents themselves are returned.
    :param float base: The base of the exponential.
    :param float minimum: Lower boundary for the exponents
    :param float maximum: Upper boundary for the exponents
    :param int | tuple of ints size: Shape of the output. See np.random.uniform for more details.
    :param bool return_exp: If False (the default), return the exponentiation of the base. If this is set to True, the
    exponents are returned.
    :return np.ndarray: Drawn samples from the distribution.
    """
    uniform_exp = np.random.uniform(minimum, maximum, size=size)
    if return_exp:
        return uniform_exp
    return np.power(base, uniform_exp)


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
