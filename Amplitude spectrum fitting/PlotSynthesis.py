import numpy as np
import matplotlib.pyplot as plt
from SaveTemplate import target_traces, target_times, target_slices, target_particles, target_x_max, TARGET_NR
from SaveTemplate import temp_times, temp_slices, temp_particles, temp_x_max, TEMPLATE_NR


TIME_DELTA = 2e-10
MIN_FREQ = 0
MAX_FREQ = 499e6

n_antenna, n_pol, n_time = target_traces.shape
synth_traces = np.loadtxt(f'{TEMPLATE_NR}_to_{TARGET_NR}_shape{(n_antenna, n_pol-1, n_time)}').\
    reshape((n_antenna, n_pol-1, n_time))

ax_x_lim = [(0, 8), (5, 20), (20, 100)]
ax_y_lim = [(-250, 650), (-100, 400), (-7.5, 10)]


def plot_antenna(axis, synth_antenna, target_antenna, antenna_number=-1, x_lim=None, y_lim=None):
    axis.plot(temp_times[antenna_number] * 1e9, synth_antenna[0], c='maroon', linestyle='--')
    axis.plot(target_times[antenna_number] * 1e9, target_antenna[0], c='k', linestyle='--')

    axis.plot(temp_times[antenna_number] * 1e9, synth_antenna[1], c='maroon', label='Synthesized')
    axis.plot(target_times[antenna_number] * 1e9, target_antenna[1], c='k', label='Real')

    axis.set_xlabel('Time [ns]')
    axis.set_title(f'Antenna number {antenna_number}')
    axis.legend()

    if x_lim is not None:
        axis.set_xlim(x_lim)

    if y_lim is not None:
        axis.set_ylim(y_lim)


def filter_pulse(pulse, f_min=MIN_FREQ, f_max=MAX_FREQ):
    freq = np.fft.rfftfreq(len(pulse), TIME_DELTA)

    spectrum = np.fft.rfft(pulse)
    filtered = spectrum * np.logical_and(freq >= f_min, freq <= f_max)

    return np.fft.irfft(filtered)


fig, ax = plt.subplots(2, 2, figsize=(10, 15))
ax = ax.flatten()

ax[0].plot(temp_slices, temp_particles, c='purple', label='Template')
ax[0].plot(target_slices, target_particles, c='k', label='Real')
ax[0].legend()

for i, antenna in enumerate([1, 3, 5]):
    synth_filtered = np.apply_along_axis(filter_pulse, 1, synth_traces[antenna, :, :])
    real_filtered = np.apply_along_axis(filter_pulse, 1, target_traces[antenna, :, :])
    plot_antenna(ax[i + 1], synth_filtered, real_filtered,
                 antenna_number=antenna, x_lim=ax_x_lim[i], y_lim=ax_y_lim[i])

fig.suptitle(r'$X_{max}^{template}$ = ' + str(int(temp_x_max)) + ' - '
             + r'$X_{max}^{real}$ = ' + str(int(target_x_max)))

plt.show()
