import os
import numpy as np
import matplotlib.pyplot as plt


SLICE = 350
ANTENNA = (1, 3, 5)

SIM_DIRECTORY = '/mnt/hgfs/Shared data/BulkSynth/bulksynth-17/'
PARAM_DIRECTORY = '/home/mdesmet/PycharmProjects/cr-template-synthesis/Amplitude spectrum fitting/paramProfileFitUB50/'
F0 = 50
TEMPLATE_NR = '100001'
TARGET_NR = '100000'


def get_d_fit(slice_grams, antenna_nr=ANTENNA):
    dist = np.array([1, 4000, 7500, 11000, 15000, 37500])
    val = [1e-9 * (slice_grams / 400 - 1.5) * np.exp(1 - antenna_dist / 40000)
           for antenna_dist in dist[list(antenna_nr)]]
    return [max(0, d) for d in val]


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


def calc_amp_corr(x_max, params, f, d=0.):
    a0 = np.polynomial.polynomial.polyval(x_max, params[antenna, :3])
    b = np.polynomial.polynomial.polyval(x_max, params[antenna, 3:6])
    c = np.polynomial.polynomial.polyval(x_max, params[antenna, 6:])

    return a0 * np.exp(b * (f - F0) + c * (f - F0) ** 2) + d


def norm_template(temp_trace, temp_amp_corr, temp_n_slice):
    spectrum = np.apply_along_axis(np.fft.rfft, 0, temp_trace)
    temp_amp, temp_phase = np.abs(spectrum), np.angle(spectrum)

    amp_res = temp_amp / temp_amp_corr / temp_n_slice

    return amp_res, temp_phase


def filter_pulse(pulse, f_range):
    filtered = pulse * f_range
    return filtered


# Get all the fit parameters from the files
param_x = np.genfromtxt(os.path.join(PARAM_DIRECTORY, 'fitX', f'slice{SLICE}.dat'))[:, 1:]
param_y = np.genfromtxt(os.path.join(PARAM_DIRECTORY, 'fitY', f'slice{SLICE}.dat'))[:, 1:]
d_fit = get_d_fit(SLICE)

# Prepare the figure
fig_amp, ax = plt.subplots(2, 2, num=1, figsize=(10, 15))
ax = ax.flatten()

# Load the shower parameters
x_slices, template_long, template_max = load_shower_params(TEMPLATE_NR)
_, target_long, target_max = load_shower_params(TARGET_NR)

# Plot the longitudinal evolution
ax[0].plot(x_slices, template_long, c='purple', label='Template')
ax[0].plot(x_slices, target_long, c='k', label='Real')
ax[0].legend()

for ind, antenna in enumerate(ANTENNA):
    template = np.genfromtxt(os.path.join(SIM_DIRECTORY, f'SIM{TEMPLATE_NR}_coreas', f'raw_{antenna}x{SLICE}.dat'))
    target = np.genfromtxt(os.path.join(SIM_DIRECTORY, f'SIM{TARGET_NR}_coreas', f'raw_{antenna}x{SLICE}.dat'))

    freq = np.fft.rfftfreq(len(template), 2e-10) / 1e6
    freq_range = freq < 502

    amp_corr_x = calc_amp_corr(template_max, param_x, freq, d_fit[ind])
    amp_corr_y = calc_amp_corr(template_max, param_y, freq, d_fit[ind])
    template_amp, template_phase = norm_template(template[:, 1:3], np.stack((amp_corr_x, amp_corr_y)).T,
                                                 template_long[int(SLICE / 5 - 1)])

    amp_corr_x = calc_amp_corr(target_max, param_x, freq)
    amp_corr_y = calc_amp_corr(target_max, param_y, freq)
    synth_amp = template_amp * np.stack((amp_corr_x, amp_corr_y)).T * target_long[int(SLICE / 5 - 1)]
    synth_phase = template_phase

    target_amp, target_phase = norm_template(target[:, 1:3], np.ones(synth_amp.shape), 1.)

    ax[ind + 1].plot(freq[freq_range], synth_amp[:, 0][freq_range], c='maroon', linestyle='--')
    ax[ind + 1].plot(freq[freq_range], target_amp[:, 0][freq_range], c='k', linestyle='--')
    ax[ind + 1].plot(freq[freq_range], synth_amp[:, 1][freq_range], c='maroon')
    ax[ind + 1].plot(freq[freq_range], target_amp[:, 1][freq_range], c='k')
    ax[ind + 1].set_title(f'Antenna {antenna}')

# Set figure parameters
fig_amp.suptitle(r'Mapping $X_{max}^{temp}$ = ' + str(int(template_max))
                 + r' to $X_{max}^{target}$ = ' + str(int(target_max))
                 + f'\n Amplitude spectra for slice {SLICE} g/cm2')
plt.show()
