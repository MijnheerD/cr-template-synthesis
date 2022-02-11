"""
Tool for automatic benchmarker of synthesis result, as a function of difference in X_max
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from GlobalVars import RANGES_LOFAR as RANGES
from GlobalVars import FIG_DIRECTORY as FIG
from GlobalVars import TIME_DELTA, DISTANCES
from crsynthesizer.FileReaderWriter import working_directory

F_MIN, F_MAX, F0 = RANGES
FIG_DIR = os.path.join(FIG, 'Benchmarks')
DISTANCES = np.array(DISTANCES)


def amplitude_function(params, frequencies, d_noise=0.):
    return params[0] * np.exp(params[1] * (frequencies - F0) + params[2] * (frequencies - F0) ** 2) + d_noise


def filter_pulse(pulse, f_min=F_MIN, f_max=F_MAX, delta=TIME_DELTA):
    freq = np.fft.rfftfreq(len(pulse), delta) / 1e6  # Freq in MHz

    spectrum = np.fft.rfft(pulse)
    filtered = spectrum * np.logical_and(freq >= f_min, freq <= f_max)

    return np.fft.irfft(filtered)


def fluence(signal, delta=TIME_DELTA):
    from scipy.constants import c, epsilon_0

    return c * epsilon_0 * delta * np.sum(np.array(signal) ** 2)


class Template:
    def __init__(self, number, source, param_dir):
        self.raw_dir = os.path.join(source, f'SIM{number}_coreas')
        self.long = os.path.join(source, f'DAT{number}.long')
        self.reas = os.path.join(source, f'SIM{number}.reas')

        self.parameters = param_dir
        self.n = self._find_dimensions()

        self.norm_amplitude = None
        self.norm_phase = None

    def _find_dimensions(self):
        x_slices, _, _ = self.load_shower_parameters()

        n_slices = len(x_slices)
        n_antennas = int(len(os.listdir(self.raw_dir)) / len(x_slices))
        n_pol = 2

        test = np.loadtxt(os.path.join(self.raw_dir, 'raw_0x5.dat'))
        n_time = test.shape[0]

        freq = np.fft.rfftfreq(n_time, TIME_DELTA) / 1e6  # Frequency in MHz
        n_freq = len(freq)

        return n_slices, n_antennas, n_pol, n_freq, n_time

    def load_shower_parameters(self):
        """
        Load the relevant parameters for the shower.
        :return: The values for the slices in g/cm^2, the sum of electrons and positrons in each slice and the X_max.
        """

        long = np.genfromtxt(self.long, skip_header=2, skip_footer=216, usecols=(0, 2, 3))

        with open(self.long, 'r') as long_file:
            for _ in range(422):
                long_file.readline()

            parameters = long_file.readline().split()
            assert parameters[0] == 'PARAMETERS', "Not on the correct line for the parameters"

            x_max = float(parameters[4])

        return long[:, 0], np.sum(long[:, 1:], axis=1), x_max

    def load_shower_pulse(self, filtered=True):
        """
        Load the radio pulse of the shower in every antenna.
        :return: Time traces in every antenna, with corresponding time values
        """
        from scipy.constants import c

        x_slices, _, _ = self.load_shower_parameters()

        with working_directory(self.raw_dir):
            times = np.zeros((self.n[1], self.n[4]))  # shape = n_antennas, n_time
            traces = np.zeros((self.n[1], self.n[2], self.n[4]))  # shape = n_antennas, n_pol, n_time

            for antenna in range(self.n[1]):
                times[antenna] = np.genfromtxt(f'raw_{antenna}x5.dat', usecols=(0,))
                antenna_data = np.zeros(traces.shape[1:])
                for x_slice in x_slices:
                    antenna_data += np.loadtxt(f'raw_{antenna}x{int(x_slice)}.dat')[:, 1:(1+self.n[2])].T * c * 1e2
                traces[antenna] = antenna_data

        if filtered:
            return np.apply_along_axis(filter_pulse, 2, traces), times

        return traces, times

    def load_fit_parameters(self):
        fit_params = np.zeros((*self.n[:3], 3, 3))

        for slice_nr in range(fit_params.shape[0]):
            with open(os.path.join(self.parameters, 'fitX', f'slice{int((slice_nr + 1) * 5)}.dat'), 'r') as fileX, \
                    open(os.path.join(self.parameters, 'fitY', f'slice{int((slice_nr + 1) * 5)}.dat'), 'r') as fileY:
                dataX = np.genfromtxt(fileX)
                dataY = np.genfromtxt(fileY)

                fit_params[slice_nr, :, 0, :, :] = dataX[:, 1:].reshape((fit_params.shape[1], 3, 3))
                fit_params[slice_nr, :, 1, :, :] = dataY[:, 1:].reshape((fit_params.shape[1], 3, 3))

        return fit_params

    def normalise(self):
        from scipy.constants import c

        temp_params = self.load_fit_parameters()
        temp_slices, temp_long, temp_x_max = self.load_shower_parameters()
        temp_long[np.where(temp_long == 0)] = 1.

        slices, _, _ = self.load_shower_parameters()
        inner = 1e-9 * (slices[np.newaxis, :] / 400 - 1.5) * np.exp(1 - DISTANCES[:, np.newaxis] / 40000)
        d = np.maximum(inner, np.zeros(inner.shape)) ** 2

        amplitude = np.zeros(self.n[:4])
        phases = np.zeros(self.n[:4])

        freq = np.fft.rfftfreq(self.n[4], TIME_DELTA) / 1e6  # Frequency in MHz
        with working_directory(self.raw_dir):
            for slice_nr in range(amplitude.shape[0]):
                for antenna_nr in range(amplitude.shape[1]):
                    data = np.genfromtxt(f'raw_{antenna_nr}x{int((slice_nr + 1) * 5)}.dat') * c * 1e2
                    spectrum = np.apply_along_axis(np.fft.rfft, 0,
                                                   data[:, 1:1 + amplitude.shape[2]] / temp_long[slice_nr]).T

                    fit_params = temp_params[slice_nr, antenna_nr]
                    amp_params = np.polynomial.polynomial.polyval(temp_x_max, fit_params.T).T
                    amp_corr = np.apply_along_axis(lambda p: amplitude_function(p, freq, d[antenna_nr, slice_nr]),
                                                   1, amp_params)
                    amp_corr[np.where(amp_corr <= 1e-32)] = 1.

                    amplitude[slice_nr, antenna_nr] = np.apply_along_axis(np.abs, 1, spectrum) / amp_corr
                    phases[slice_nr, antenna_nr] = np.apply_along_axis(np.angle, 1, spectrum)

        self.norm_amplitude = amplitude
        self.norm_phase = phases

        return amplitude, phases

    def map_to_target(self, target):
        """

        :param Template target: target showers
        :return:
        """
        synth = np.zeros((self.n[1], self.n[2], self.n[4]))

        target_params = target.load_fit_parameters()
        target_slices, target_long, target_x_max = target.load_shower_parameters()

        freq = np.fft.rfftfreq(self.n[4], TIME_DELTA) / 1e6  # Frequency in MHz
        f_range = np.logical_and(freq > F_MIN, freq < F_MAX)

        for slice_nr in range(self.n[0]):
            for antenna_nr in range(self.n[1]):
                fit_params = target_params[slice_nr, antenna_nr]
                amp_params = np.polynomial.polynomial.polyval(target_x_max, fit_params.T).T

                target_amp = np.apply_along_axis(lambda p: amplitude_function(p, freq), 1, amp_params)

                synth_spectrum = self.norm_amplitude[slice_nr, antenna_nr] * target_amp * target_long[slice_nr] * \
                    np.exp(1j * self.norm_phase[slice_nr, antenna_nr])
                synth_spectrum[:, np.logical_not(f_range)] = 0.
                synth[antenna_nr, :, :] += np.apply_along_axis(np.fft.irfft, 1, synth_spectrum)

        return synth


class Benchmark:
    def __init__(self, showers_dir, param_dir, mode="peak_ratio"):
        if isinstance(showers_dir, str):
            self.showers = list([showers_dir])
        else:
            self.showers = showers_dir
        self.parameters = param_dir

        self.mode = mode
        self.scores = {}

    def select_showers(self, k=20):
        from random import sample
        showers = []

        for path in self.showers:
            with working_directory(path):
                showers.extend([Template(name.split('_')[0][3:], path, self.parameters)
                                for name in sample(glob.glob("SIM*_coreas"), k)])

        return showers

    def score_synth(self, synth, real):
        if self.mode == 'peak_ratio':
            return np.amax(synth, axis=2) / np.amax(real, axis=2)
        elif self.mode == 'peak_diff':
            return np.amax(synth) - np.amax(real)
        elif self.mode == 'energy_ratio':
            return fluence(synth) / fluence(real)
        elif self.mode == 'energy_diff':
            return fluence(synth) - fluence(real)
        elif self.mode == 'correlation_max':
            from scipy.signal import correlate, correlation_lags

            lags = correlation_lags(len(synth), len(real))
            window = np.logical_and(lags > -100, lags < 100)
            return np.amax(correlate(synth, real)[window])
        elif self.mode == 'correlation_shift':
            from scipy.signal import correlate, correlation_lags

            lags = correlation_lags(len(synth), len(real))
            window = np.logical_and(lags > -100, lags < 100)
            return np.argmax(correlate(synth, real)[window])
        elif self.mode == 'correlation_center':
            from scipy.signal import correlate, correlation_lags

            lags = correlation_lags(len(synth), len(real))
            window = np.where(lags == 0)
            return correlate(synth, real)[window]

    def run(self, number=4):
        templates = self.select_showers(k=number)
        targets = self.select_showers(k=number)

        for template in templates:
            template.normalise()

            for target in targets:
                synth = template.map_to_target(target)
                real = target.load_shower_pulse()[0]

                _, _, synth_max = template.load_shower_parameters()
                _, _, real_max = target.load_shower_parameters()

                self.scores[(synth_max, real_max)] = self.score_synth(synth, real)

    def plot_scores(self):
        import matplotlib.cm as cm
        import matplotlib.colors as colors

        scores = list(self.scores.values())
        temp_max, real_max = zip(*self.scores.keys())

        n_antenna, n_pol = scores[0].shape

        for ant in range(n_antenna):
            antenna_scores = np.array([score[ant] for score in scores])

            cmap = cm.plasma
            norm = colors.Normalize(vmin=np.amin(antenna_scores), vmax=np.amax(antenna_scores))

            fig, ax = plt.subplots(1, n_pol, num=ant, figsize=(12, 6))
            for pol in range(n_pol):
                ax[pol].scatter(temp_max, real_max, c=antenna_scores[:, pol], cmap=cmap, norm=norm)
                ax[pol].plot([min(temp_max + real_max), max(temp_max + real_max)],
                             [min(temp_max + real_max), max(temp_max + real_max)],
                             c='maroon', linestyle='--', linewidth=0.5)

                ax[pol].set_xlabel(r"$X_{max}$ of template")
                ax[pol].set_ylabel(r"$X_{max}$ of target")

                ax[pol].set_aspect("equal")

            fig.suptitle(f'{self.mode} benchmark for antenna {ant}')
            fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax[:], location='right', shrink=0.7)
            fig.savefig(os.path.join(FIG_DIR, f'benchmark_{self.mode}_antenna{ant}.png'), bbox_inches='tight')
