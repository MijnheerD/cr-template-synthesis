import os
import sys
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__" and len(sys.argv) > 1:
    SLICE = int(sys.argv[1])
else:
    SLICE = 700
ANTENNA = (1, 3, 5)
X_LIM = [(0, 10), (2, 16), (50, 70)]

SIM_DIRECTORY = '/mnt/hgfs/Shared data/BulkSynth/bulksynth-17/'
FIG_DIRECTORY = '../Figures/SpectrumComparisons'
PARAM_DIRECTORY = '../Parameters/paramProfileFit_LOFAR/'
F0 = 50
TEMPLATE_NR = '100070'
TARGET_NR = '100031'


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


def load_fit_parameters_david(slice_nr):
    file = '/home/mdesmet/PycharmProjects/davidMWE/ampfit_pars_shape(2, 1, 207, 6, 2, 3, 3)'
    fit_params = np.loadtxt(file).reshape((2, 1, 207, 6, 2, 3, 3))

    params_x = np.zeros((6, 9))
    params_y = np.zeros((6, 9))

    params_x[:, :3] = fit_params[0, 0, slice_nr, :, 0, 0, :]
    params_x[:, 3:6] = fit_params[0, 0, slice_nr, :, 0, 1, :]
    params_x[:, 6:] = fit_params[0, 0, slice_nr, :, 0, 2, :]

    params_y[:, :3] = fit_params[0, 0, slice_nr, :, 1, 0, :]
    params_y[:, 3:6] = fit_params[0, 0, slice_nr, :, 1, 1, :]
    params_y[:, 6:] = fit_params[0, 0, slice_nr, :, 1, 2, :]

    return params_x, params_y


def calc_amp_corr(x_max, params, f, d=0., f0=F0):
    a0 = np.polynomial.polynomial.polyval(x_max, params[antenna, :3])
    b = np.polynomial.polynomial.polyval(x_max, params[antenna, 3:6])
    c = np.polynomial.polynomial.polyval(x_max, params[antenna, 6:])

    return a0 * np.exp(b * (f - f0) + c * (f - f0) ** 2) + d


def norm_template(temp_trace, temp_amp_corr, temp_n_slice):
    spectrum = np.apply_along_axis(np.fft.rfft, 0, temp_trace)
    temp_amp, temp_phase = np.abs(spectrum), np.angle(spectrum)

    amp_res = temp_amp / temp_amp_corr / temp_n_slice

    return amp_res, temp_phase


def filter_pulse(pulse, f_range):
    spectrum = np.fft.rfft(pulse)
    filtered = spectrum * f_range

    return np.fft.irfft(filtered)


# Get all the fit parameters from the files
param_x = np.genfromtxt(os.path.join(PARAM_DIRECTORY, 'fitX', f'slice{SLICE}.dat'))[:, 1:]
param_y = np.genfromtxt(os.path.join(PARAM_DIRECTORY, 'fitY', f'slice{SLICE}.dat'))[:, 1:]
param_x_david, param_y_david = load_fit_parameters_david(int(SLICE / 5 - 1))
d_fit = get_d_fit(SLICE)

# Prepare the figure
fig_amp, ax = plt.subplots(2, 2, num=1, figsize=(15, 10))
fig_phase, ax2 = plt.subplots(2, 2, num=2, figsize=(12, 10))
fig_pulse, ax3 = plt.subplots(2, 2, num=3, figsize=(16, 12))
ax = ax.flatten()
ax2 = ax2.flatten()
ax3 = ax3.flatten()

# Load the shower parameters
x_slices, template_long, template_max = load_shower_params(TEMPLATE_NR)
_, target_long, target_max = load_shower_params(TARGET_NR)

# Plot the longitudinal evolution
ax[0].plot(x_slices, template_long, c='purple', label='Template')
ax[0].plot(x_slices, target_long, c='k', label='Real')
ax[0].legend()

ax2[0].plot(x_slices, template_long, c='purple', label='Template')
ax2[0].plot(x_slices, target_long, c='k', label='Real')
ax2[0].legend()

ax3[0].plot(x_slices, template_long, c='purple', label='Template')
ax3[0].plot(x_slices, target_long, c='k', label='Real')
ax3[0].legend()

for ind, antenna in enumerate(ANTENNA):
    template = np.genfromtxt(os.path.join(SIM_DIRECTORY, f'SIM{TEMPLATE_NR}_coreas', f'raw_{antenna}x{SLICE}.dat'))
    target = np.genfromtxt(os.path.join(SIM_DIRECTORY, f'SIM{TARGET_NR}_coreas', f'raw_{antenna}x{SLICE}.dat'))

    template_time = template[:, 0] * 1e9
    target_time = target[:, 0] * 1e9

    freq = np.fft.rfftfreq(len(template), 2e-10) / 1e6
    freq_range = np.logical_and(30 < freq,  freq < 80)

    # Calculate the normalised template using my parameters
    amp_corr_x = calc_amp_corr(template_max, param_x, freq, d_fit[ind])
    amp_corr_y = calc_amp_corr(template_max, param_y, freq, d_fit[ind])
    template_amp, template_phase = norm_template(template[:, 1:3],
                                                 np.stack((amp_corr_x, amp_corr_y)).T,
                                                 template_long[int(SLICE / 5 - 1)])

    # Calculate the normalised template using David's parameters
    amp_corr_x_david = calc_amp_corr(template_max, param_x_david, freq, d_fit[ind], f0=0)
    amp_corr_y_david = calc_amp_corr(template_max, param_y_david, freq, d_fit[ind], f0=0)
    template_amp_david, template_phase_david = norm_template(template[:, 1:3],
                                                             np.stack((amp_corr_x_david, amp_corr_y_david)).T,
                                                             template_long[int(SLICE / 5 - 1)])

    # Calculate synthesized amplitude and phase using my parameters
    amp_corr_x = calc_amp_corr(target_max, param_x, freq)
    amp_corr_y = calc_amp_corr(target_max, param_y, freq)
    synth_amp = template_amp * np.stack((amp_corr_x, amp_corr_y)).T * target_long[int(SLICE / 5 - 1)]
    synth_phase = template_phase

    # Synthesize the pulse
    synth_filtered = np.apply_along_axis(np.fft.irfft, 0,
                                         synth_amp * np.exp(1j * synth_phase) * freq_range[:, np.newaxis])

    # Calculate synthesized amplitude and phase using David's parameters
    amp_corr_x_david = calc_amp_corr(target_max, param_x_david, freq, f0=0)
    amp_corr_y_david = calc_amp_corr(target_max, param_y_david, freq, f0=0)
    synth_amp_david = template_amp_david * np.stack((amp_corr_x_david, amp_corr_y_david)).T \
        * target_long[int(SLICE / 5 - 1)]
    synth_phase_david = template_phase_david

    # Synthesize the pulse with David's amplitude and phases
    synth_filtered_david = np.apply_along_axis(np.fft.irfft, 0,
                                               synth_amp_david * np.exp(1j * synth_phase_david) *
                                               freq_range[:, np.newaxis])

    target_amp, target_phase = norm_template(target[:, 1:3], np.ones(synth_amp.shape), 1.)

    # Make a quadratic fit of the real amplitude spectrum in the freq range
    poly_charge = np.polyfit(freq[freq_range], np.log(target_amp[:, 0][freq_range]), 2)
    poly_geo = np.polyfit(freq[freq_range], np.log(target_amp[:, 1][freq_range]), 2)

    # Plot the amplitude for each antenna with fit overlayed
    ax[ind + 1].plot(freq[freq_range], synth_amp[:, 0][freq_range], c='maroon', linestyle='--')
    ax[ind + 1].plot(freq[freq_range], synth_amp_david[:, 0][freq_range], c='green', linestyle='--')
    ax[ind + 1].plot(freq[freq_range], target_amp[:, 0][freq_range], c='k', linestyle='--')
    ax[ind + 1].plot(freq[freq_range], np.exp(np.polyval(poly_charge, freq[freq_range])),
                     c='mediumblue', linestyle='--')
    ax[ind + 1].plot(freq[freq_range], synth_amp[:, 1][freq_range], c='maroon', label='Synthesized')
    ax[ind + 1].plot(freq[freq_range], synth_amp_david[:, 1][freq_range], c='green', label='David parameters')
    ax[ind + 1].plot(freq[freq_range], target_amp[:, 1][freq_range], c='k', label='Real')
    ax[ind + 1].plot(freq[freq_range], np.exp(np.polyval(poly_geo, freq[freq_range])),
                     c='mediumblue', label='Fit to real')
    ax[ind + 1].set_xlabel(r'Freq $[MHz]$')
    ax[ind + 1].set_yscale('log')
    ax[ind + 1].set_title(f'Antenna {antenna}')
    ax[ind + 1].legend()

    # Plot the phase spectrum for each antenna
    ax2[ind + 1].plot(freq[freq_range], synth_phase[:, 0][freq_range], c='maroon', linestyle='--')
    ax2[ind + 1].plot(freq[freq_range], target_phase[:, 0][freq_range], c='k', linestyle='--')
    ax2[ind + 1].plot(freq[freq_range], synth_phase[:, 1][freq_range], c='maroon', label='Synthesized')
    ax2[ind + 1].plot(freq[freq_range], target_phase[:, 1][freq_range], c='k', label='Real')
    ax2[ind + 1].set_xlim([0, 100])
    ax2[ind + 1].set_xlabel(r'Freq $[MHz]$')
    ax2[ind + 1].set_title(f'Antenna {antenna}')
    ax2[ind + 1].legend()

    # Plot pulse shapes in each antenna
    ax3[ind + 1].plot(template_time, synth_filtered[:, 0], c='maroon', linestyle='--')
    ax3[ind + 1].plot(template_time, synth_filtered_david[:, 0], c='green', linestyle='--')
    ax3[ind + 1].plot(target_time, filter_pulse(target[:, 1], freq_range), c='k', linestyle='--')
    ax3[ind + 1].plot(template_time, synth_filtered[:, 1], c='maroon', label='Synthesized')
    ax3[ind + 1].plot(template_time, synth_filtered_david[:, 1], c='green', label='David parameters')
    ax3[ind + 1].plot(target_time, filter_pulse(target[:, 2], freq_range), c='k', label='Real')
    ax3[ind + 1].set_xlim(X_LIM[ind])
    ax3[ind + 1].set_xlabel(r'Time $[ns]$')
    ax3[ind + 1].set_title(f'Antenna {antenna}')
    ax3[ind + 1].legend()

# Set figure parameters and save
fig_amp.suptitle(r'Mapping $X_{max}^{temp}$ = ' + str(int(template_max))
                 + r' to $X_{max}^{target}$ = ' + str(int(target_max))
                 + f'\n Amplitude spectra for slice {SLICE} g/cm2')
fig_amp.savefig(os.path.join(FIG_DIRECTORY, f'AmpSpectrumComparison{SLICE}.png'), bbox_inches='tight')

fig_phase.suptitle(r'Mapping $X_{max}^{temp}$ = ' + str(int(template_max))
                   + r' to $X_{max}^{target}$ = ' + str(int(target_max))
                   + f'\n Phase spectra for slice {SLICE} g/cm2')
fig_phase.savefig(os.path.join(FIG_DIRECTORY, f'PhiSpectrumComparison{SLICE}.png'), bbox_inches='tight')

fig_pulse.suptitle(r'Mapping $X_{max}^{temp}$ = ' + str(int(template_max))
                   + r' to $X_{max}^{target}$ = ' + str(int(target_max))
                   + f'\n Pulses for slice {SLICE} g/cm2')
fig_pulse.savefig(os.path.join(FIG_DIRECTORY, f'PulseComparison{SLICE}.png'), bbox_inches='tight')
