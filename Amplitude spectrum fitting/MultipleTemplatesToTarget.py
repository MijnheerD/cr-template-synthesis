from GlobalVars import SIM_DIRECTORY as SIMS
from GlobalVars import PARAM_DIRECTORY_LOFAR as PARAM_DIR
from GlobalVars import RANGES_LOFAR as RANGES
from GlobalVars import FIG_DIRECTORY

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c as c_vacuum


SIM_DIR = SIMS[0]
FIG_DIR = os.path.join(FIG_DIRECTORY, 'TempToTarget')
F_MIN, F_MAX, F0 = RANGES
TARGET_NR = '100000'

TEMPLATE_X_MAX = 600
NR_OF_TEMPLATES = 2

TIMES = [(0, 15), (0, 15), (0, 20), (0, 25), (0, 50), (0, 100)]  # ns


def amplitude_function(params, frequencies, d_noise=0.):
    return params[0] * np.exp(params[1] * (frequencies - F0) + params[2] * (frequencies - F0) ** 2) + d_noise


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


def norm_template(temp_nr, temp_long, temp_max, temp_params, path=SIM_DIR):
    temp_dir = os.path.join(path, f'SIM{temp_nr}_coreas')
    temp_long[np.where(temp_long == 0)] = 1.

    amplitude = np.zeros((n_slice, n_antenna, n_pol, n_freq))
    phases = np.zeros((n_slice, n_antenna, n_pol, n_freq))

    prev = os.getcwd()
    os.chdir(temp_dir)
    for slice_nr in range(n_slice):
        for antenna_nr in range(n_antenna):
            data = np.genfromtxt(f'raw_{antenna_nr}x{int((slice_nr + 1) * 5)}.dat') * c_vacuum * 1e2
            spectrum = np.apply_along_axis(np.fft.rfft, 0, data[:, 1:1+n_pol] / temp_long[slice_nr]).T

            fit_params = temp_params[antenna_nr, slice_nr]
            amp_params = np.polynomial.polynomial.polyval(temp_max, fit_params.T).T
            amp_corr = np.apply_along_axis(lambda p: amplitude_function(p, freq, d[antenna_nr, slice_nr]),
                                           1, amp_params)
            amp_corr[np.where(amp_corr <= 1e-32)] = 1.

            amplitude[slice_nr, antenna_nr] = np.apply_along_axis(np.abs, 1, spectrum) / amp_corr
            phases[slice_nr, antenna_nr] = np.apply_along_axis(np.angle, 1, spectrum)

    os.chdir(prev)

    return amplitude, phases


def synthesize_shower(temp_amps, temp_phase, target_long, target_max, target_params):
    synth = np.zeros((n_antenna, n_pol, n_time))

    for slice_nr in range(n_slice):
        for antenna_nr in range(n_antenna):
            fit_params = target_params[antenna_nr, slice_nr]
            amp_params = np.polynomial.polynomial.polyval(target_max, fit_params.T).T

            target_amp = np.apply_along_axis(lambda p: amplitude_function(p, freq), 1, amp_params)

            synth_spectrum = temp_amps[slice_nr, antenna_nr] * target_amp * target_long[slice_nr] * \
                np.exp(1j * temp_phase[slice_nr, antenna_nr])
            synth_spectrum[:, np.logical_not(f_range)] = 0.
            synth[antenna_nr, :, :] += np.apply_along_axis(np.fft.irfft, 1, synth_spectrum)

    return synth


def filter_pulse(pulse, f_min=F_MIN, f_max=F_MAX):
    spectrum = np.fft.rfft(pulse)
    filtered = spectrum * np.logical_and(freq >= f_min, freq <= f_max)

    return np.fft.irfft(filtered)


def load_fit_parameters(path=PARAM_DIR):
    fit_params = np.zeros((n_antenna, n_slice, n_pol, 3, 3))

    for slice_nr in range(n_slice):
        with open(os.path.join(path, 'fitX', f'slice{int((slice_nr + 1) * 5)}.dat'), 'r') as fileX, \
                open(os.path.join(path, 'fitY', f'slice{int((slice_nr + 1) * 5)}.dat'), 'r') as fileY:
            dataX = np.genfromtxt(fileX)
            dataY = np.genfromtxt(fileY)

            fit_params[:, slice_nr, 0, :, :] = dataX[:, 1:].reshape((n_antenna, 3, 3))
            fit_params[:, slice_nr, 1, :, :] = dataY[:, 1:].reshape((n_antenna, 3, 3))

    return fit_params


def add_shower_to_figs(time, shower, axs, color=None, label=None):
    assert shower.shape[0] == len(axs), "Not enough axes provided"

    if label is None:
        label = 'Template'

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


count = 0
template = 100000
suitable_templates = {}

# Loop over all proton templates and check whether they match the X_max condition
while count < NR_OF_TEMPLATES and template < 100100:
    temp_slices, temp_particles, temp_x_max = load_shower_params(template)
    if abs(TEMPLATE_X_MAX - temp_x_max) < 10:
        count += 1
        suitable_templates[str(template)] = temp_x_max

    template += 1

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


# Load the target shower and it to the figures
target_slices, target_particles, target_x_max = load_shower_params(TARGET_NR)
target_traces, target_times = load_shower_pulse(TARGET_NR)

# Find all the quantities necessary to construct the arrays
n_slice = len(target_slices)
n_antenna, _, n_time = target_traces.shape
n_pol = 2

freq = np.fft.rfftfreq(n_time, 2e-10) / 1e6  # Frequency in MHz
f_range = np.logical_and(freq > F_MIN, freq < F_MAX)
n_freq = len(freq)

# Calculate the array containing the d values
distances = np.array([1, 4000, 7500, 11000, 15000, 37500])
slices = np.array([(ind + 1) * 5 for ind in range(n_slice)])

inner = 1e-9 * (slices[np.newaxis, :] / 400 - 1.5) * np.exp(1 - distances[:, np.newaxis] / 40000)
d = np.maximum(inner, np.zeros(inner.shape)) ** 2

# Load the fit parameters
fit_parameters = load_fit_parameters()

# Add target to the figures
add_long_to_figs(target_slices, target_particles, ax_long, color='k', label='Real')
add_shower_to_figs(target_times, target_traces, ax, color='k', label='Real')

# Take all the suitable templates and synthesize the target pulse with them
for ind, temp in enumerate(suitable_templates.keys()):
    _, temp_particles, temp_x_max = load_shower_params(temp)
    _, temp_times = load_shower_pulse(temp)
    temp_res_amp, temp_res_phi = norm_template(temp, temp_particles, temp_x_max, fit_parameters)
    synth_shower = synthesize_shower(temp_res_amp, temp_res_phi, target_particles, target_x_max, fit_parameters)

    # Add particle evolution to axes
    add_long_to_figs(target_slices, temp_particles, ax_long, label=f'Template {ind + 1}')

    # Add synthesized shower to axes
    add_shower_to_figs(temp_times, synth_shower, ax, label=f'Template {ind + 1}')

# Save all the figures
os.chdir(FIG_DIR)

for ind, figure in enumerate(fig):
    ax[ind].legend()
    ax[ind].set_xlim(TIMES[ind])
    ax[ind].set_xlabel('Time [ns]')
    ax[ind].set_ylabel(r'E [$\mu$ V/m]')
    ax[ind].set_title(f'Antenna {ind} - r = {distances[ind] / 100}m')

    ax_long[ind].set_xlabel('X [g/cm²]')
    ax_long[ind].set_ylabel('N(X)')

    figure.suptitle(f'Templates with X_max = {list(suitable_templates.values())}')
    figure.savefig(f'{len(suitable_templates)}_templates_{TEMPLATE_X_MAX}_to_{target_x_max}_antenna_{ind}.png',
                   bbox_inches='tight')
