from GlobalVars import SIM_DIRECTORY as SIMS
from GlobalVars import TIME_DELTA, DISTANCES
from GlobalVars import RANGES_LOFAR as RANGES
from GlobalVars import FIG_DIRECTORY

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c as c_vacuum


SIM_DIR = SIMS[0]
FIG_DIR = os.path.join(FIG_DIRECTORY, 'ShowersInRange')
F_MIN, F_MAX, F0 = RANGES

TARGET_X_MAX = 768
NR_OF_SHOWERS = 10

TIMES = [(0, 15), (0, 15), (0, 20), (0, 25), (0, 50), (0, 100)]  # ns


def load_shower_params(shower_nr, path=SIM_DIR):
    """
    Load the relevant parameters for a single shower.
    :return: The values for the slices in g/cm^2, the sum of electrons and positrons in each slice and the X_max.
    """
    long = np.genfromtxt(os.path.join(path, f'DAT{shower_nr}.long'), skip_header=2, skip_footer=216, usecols=(0, 2, 3))

    with open(os.path.join(path, f'DAT{shower_nr}.long'), 'r') as long_file:
        for _ in range(422):
            long_file.readline()

        parameters = long_file.readline().split()
        assert parameters[0] == 'PARAMETERS', "Not on the correct line for the parameters"

        x_max = float(parameters[4])

    return long[:, 0], np.sum(long[:, 1:], axis=1), x_max


def load_shower_pulse(shower_nr, path=SIM_DIR):
    x_slices, _, _ = load_shower_params(shower_nr)

    prev = os.getcwd()
    os.chdir(os.path.join(path, f'SIM{shower_nr}_coreas/'))

    n_antennas = int(len(os.listdir('.')) / len(x_slices))

    test = np.loadtxt('raw_0x5.dat')
    times = np.zeros((n_antennas, test.shape[0]))  # shape = n_antennas, n_time
    traces = np.zeros((n_antennas, test.shape[1] - 1, test.shape[0]))  # shape = n_antennas, n_pol, n_time

    for antenna in range(n_antennas):
        times[antenna] = np.genfromtxt(f'raw_{antenna}x5.dat', usecols=(0,))
        antenna_data = np.zeros(traces.shape[1:])
        for x_slice in x_slices:
            antenna_data += np.loadtxt(f'raw_{antenna}x{int(x_slice)}.dat')[:, 1:].T * c_vacuum * 1e2
        traces[antenna] = antenna_data

    os.chdir(prev)

    return traces, times


def filter_pulse(pulse, f_min=F_MIN, f_max=F_MAX):
    freq = np.fft.rfftfreq(len(pulse), TIME_DELTA) / 1e6  # MHz

    spectrum = np.fft.rfft(pulse)
    filtered = spectrum * np.logical_and(freq >= f_min, freq <= f_max)

    return np.fft.irfft(filtered)


def add_shower_to_figs(time, shower, axs, color=None, label=None):
    assert shower.shape[0] == len(axs), "Not enough axes provided"

    if label is None:
        label = 'Shower'

    for ind, axis in enumerate(axs):
        if color is not None:
            # axis.plot(time[ind] * 1e9, shower[ind, 0], linestyle='--', c=color)
            axis.plot(time[ind] * 1e9, filter_pulse(shower[ind, 1]), label=label, c=color)
        else:
            # axis.plot(time[ind] * 1e9, shower[ind, 0], linestyle='--')
            axis.plot(time[ind] * 1e9, filter_pulse(shower[ind, 1]), label=label)


def add_long_to_figs(x_slices, particles, axs, color=None, label=None):
    if label is None:
        label = 'Template'

    for ind, axis in enumerate(axs):
        if color is not None:
            # axis.plot(slices, shower[ind, 0], linestyle='--', c=color)
            axis.plot(x_slices, particles, label=label, c=color)
        else:
            # axis.plot(slices, shower[ind, 0], linestyle='--')
            axis.plot(x_slices, particles, label=label)


# Prepare the figures
fig0, ax0 = plt.subplots(1, 2, figsize=(12, 8))
fig1, ax1 = plt.subplots(1, 2, figsize=(12, 8))
fig2, ax2 = plt.subplots(1, 2, figsize=(12, 8))
fig3, ax3 = plt.subplots(1, 2, figsize=(12, 8))
fig4, ax4 = plt.subplots(1, 2, figsize=(12, 8))
fig5, ax5 = plt.subplots(1, 2, figsize=(12, 8))

fig = [fig0, fig1, fig2, fig3, fig4, fig5]
ax_long = [ax0[0], ax1[0], ax2[0], ax3[0], ax4[0], ax5[0]]
ax = [ax0[1], ax1[1], ax2[1], ax3[1], ax4[1], ax5[1]]

# Loop over all proton templates and check whether they match the X_max condition
count = 0
shower_select = 100000
suitable_showers = {}

while count < NR_OF_SHOWERS and shower_select < 100100:
    shower_slices, shower_particles, shower_x_max = load_shower_params(shower_select)
    if abs(TARGET_X_MAX - shower_x_max) < 10:
        count += 1
        suitable_showers[str(shower_select)] = shower_x_max

        # Add the shower to the figures
        shower_traces, shower_times = load_shower_pulse(shower_select)
        add_long_to_figs(shower_slices, shower_particles, ax_long, label=f'Shower {count}')
        add_shower_to_figs(shower_times, shower_traces, ax, label=f'Shower {count}')

    shower_select += 1

# Save all the figures
os.chdir(FIG_DIR)

for i, figure in enumerate(fig):
    ax[i].legend()
    ax[i].set_xlim(TIMES[i])
    ax[i].set_xlabel('Time [ns]')
    ax[i].set_ylabel(r'E [$\mu$ V/m]')
    ax[i].set_title(f'Antenna {i} - r = {DISTANCES[i] / 100}m')

    ax_long[i].set_xlabel('X [g/cm²]')
    ax_long[i].set_ylabel('N(X)')

    figure.suptitle(f'Showers with X_max = {list(suitable_showers.values())}')
    figure.savefig(f'{TARGET_X_MAX}_showers_antenna_{i}.png',
                   bbox_inches='tight')
