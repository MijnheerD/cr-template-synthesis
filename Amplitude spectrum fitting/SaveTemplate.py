import os
import numpy as np
from scipy.constants import c as c_vacuum

SIM_DIRECTORY = '/mnt/hgfs/Shared data/BulkSynth/bulksynth-17/'
PARAM_DIRECTORY = '/home/mdesmet/PycharmProjects/cr-template-synthesis/Amplitude spectrum fitting/paramProfileFitNew50/'
F0 = 50
TEMPLATE_NR = '100001'
TARGET_NR = '100000'


def amplitude_function(params, frequencies, d_noise=0.):
    return params[0] * np.exp(params[1] * (frequencies - F0) + params[2] * (frequencies - F0) ** 2) + d_noise ** 2


def load_shower_params(shower_nr, path=SIM_DIRECTORY):
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


def load_shower_pulse(shower_nr, path=SIM_DIRECTORY):
    x_slices, _, _ = load_shower_params(shower_nr)

    prev = os.getcwd()
    os.chdir(os.path.join(path, f'SIM{shower_nr}_coreas/'))

    n_antennas = int(len(os.listdir('.')) / len(x_slices))

    test = np.loadtxt('raw_0x5.dat')
    times = np.zeros((n_antennas, test.shape[0]))
    traces = np.zeros((n_antennas, test.shape[1] - 1, test.shape[0]))  # shape = n_antennas, n_pol, n_time

    for antenna in range(n_antennas):
        times[antenna] = np.genfromtxt(f'raw_{antenna}x5.dat', usecols=(0,))
        antenna_data = np.zeros(traces.shape[1:])
        for x_slice in x_slices:
            antenna_data += np.loadtxt(f'raw_{antenna}x{int(x_slice)}.dat')[:, 1:].T * c_vacuum * 1e2
        traces[antenna] = antenna_data

    os.chdir(prev)

    return traces, times


def norm_template(temp_nr, temp_long, temp_max, temp_params, path=SIM_DIRECTORY):
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

            amplitude[slice_nr, antenna_nr] = np.abs(spectrum) / amp_corr
            phases[slice_nr, antenna_nr] = np.angle(spectrum)

    os.chdir(prev)

    return amplitude, phases


def load_fit_parameters(path=PARAM_DIRECTORY):
    fit_params = np.zeros((n_antenna, n_slice, n_pol, 3, 3))

    for slice_nr in range(n_slice):
        with open(os.path.join(path, 'fitX', f'slice{int((slice_nr + 1) * 5)}.dat'), 'r') as fileX, \
                open(os.path.join(path, 'fitY', f'slice{int((slice_nr + 1) * 5)}.dat'), 'r') as fileY:
            dataX = np.genfromtxt(fileX)
            dataY = np.genfromtxt(fileY)

            fit_params[:, slice_nr, 0, :, :] = dataX[:, 1:].reshape((n_antenna, 3, 3))
            fit_params[:, slice_nr, 1, :, :] = dataY[:, 1:].reshape((n_antenna, 3, 3))

    return fit_params


def load_fit_parameters_david():
    file = '/home/mdesmet/PycharmProjects/davidMWE/ampfit_pars_shape(2, 1, 207, 6, 2, 3, 3)'
    fit_params = np.loadtxt(file).reshape((2, 1, 207, 6, 2, 3, 3))
    fit_params = np.swapaxes(fit_params, 2, 3)

    return fit_params[0, 0]


def synthesize_shower(temp_amps, temp_phase, target_long, target_max, target_params):
    synth = np.zeros((n_antenna, n_pol, n_time))

    for slice_nr in range(n_slice):
        for antenna_nr in range(n_antenna):
            fit_params = target_params[antenna_nr, slice_nr]
            amp_params = np.polynomial.polynomial.polyval(target_max, fit_params.T).T

            target_amp = np.apply_along_axis(lambda p: amplitude_function(p, freq), 1, amp_params)
            target_amp[:, np.logical_not(f_range)] = 0.

            synth_spectrum = temp_amps[slice_nr, antenna_nr] * target_amp * target_long[slice_nr] * \
                             np.exp(1j * temp_phase[slice_nr, antenna_nr])
            synth[antenna_nr, :, :] += np.apply_along_axis(np.fft.irfft, 1, synth_spectrum)

    return synth


# Template and target shower parameters
temp_slices, temp_particles, temp_x_max = load_shower_params(TEMPLATE_NR)
target_slices, target_particles, target_x_max = load_shower_params(TARGET_NR)

# Load showers for convenience
temp_traces, temp_times = load_shower_pulse(TEMPLATE_NR)
target_traces, target_times = load_shower_pulse(TARGET_NR)

# Find all the quantities necessary to construct the arrays
n_slice = len(target_slices)
n_antenna, _, n_time = target_traces.shape
n_pol = 2

freq = np.fft.rfftfreq(n_time, 2e-10) / 1e6  # Frequency in MHz
f_range = freq < 500
n_freq = len(freq)

if __name__ == '__main__':
    # Calculate the array containing the d values
    distances = np.array([1, 4000, 7500, 11000, 15000, 37500])
    slices = np.array([(ind + 1) * 5 for ind in range(n_slice)])

    inner = 1e-9 * (slices[np.newaxis, :] / 400 - 1.5) * np.exp(1 - distances[:, np.newaxis] / 40000)
    d = np.maximum(inner, np.zeros(inner.shape)) ** 2

    # Analyze all the files of the template shower
    fit_parameters = load_fit_parameters()
    temp_res_amp, temp_res_phi = norm_template(TEMPLATE_NR, temp_particles, temp_x_max, fit_parameters)

    # Synthesize and save target shower
    synth_shower = synthesize_shower(temp_res_amp, temp_res_phi, target_particles, target_x_max, fit_parameters)
    to_save = synth_shower[:, :2, :]  # / c_vacuum / 1e2
    np.savetxt(f'{TEMPLATE_NR}_to_{TARGET_NR}_shape{to_save.shape}', to_save.flatten())
