import numpy as np
from scipy.optimize import curve_fit
from NumpyEncoder import write_file_json, read_file_json


class Fitter(object):
    def __init__(self, func):
        # Parameters related to the fitting
        self.fit_fun = func
        self.p0 = []
        self.sigma = 0
        self.bounds = (-np.inf, np.inf)
        # Parameters to store best fit results
        self.opt = []
        self.cov = []

    def fit(self, x_data, y_data):
        """
        Fits a two-dimensional array along the first axis, using x_data as the x-values for every slice. The best fit
        parameters for every slice along the second dimension are stored along the same dimension in self.opt.
        :param np.ndarray | list x_data:
        :param np.ndarray y_data:
        """
        for pol in range(y_data.shape[1]):
            res = curve_fit(self.fit_fun, x_data, y_data[:, pol],
                            p0=self.p0, sigma=self.sigma[:, pol], bounds=self.bounds)
            self.opt.append(res[0])
            self.cov.append(np.sqrt(np.diag(res[1])))
        self.opt = np.array(self.opt).T
        self.cov = np.array(self.cov).T

    def fit_multi(self, x_data, y_data):
        assert type(self.fit_fun)==list, "Fit function is not a list"

        assert len(self.fit_fun)==y_data.shape[1], "Not enough fit functions to cover all polarizations"

        for ind in range(len(self.fit_fun)):
            res = curve_fit(self.fit_fun[ind], x_data, y_data[:, ind],
                            p0=self.p0[ind], sigma=self.sigma[:, ind], bounds=self.bounds[ind])
            self.opt.append(res[0])
            self.cov.append(np.sqrt(np.diag(res[1])))
        self.opt = np.array(self.opt).T
        self.cov = np.array(self.cov).T


class AmplitudeFitter(Fitter):
    """
    This object fits the amplitude in a given frequency range. The first dimension of the trace argument is taken to be
    the time axis and the second dimension should correspond to the different polarizations.
    """
    def __init__(self, trace: np.array, fit_range: tuple, d=0, f0=0):
        super().__init__(lambda f, a_0, b, c: a_0 * np.exp(b * (f - f0) + c * (f - f0) ** 2) + d)

        self.f0 = f0
        self.d = d
        self.time_trace = trace if trace.ndim == 2 else trace[:, np.newaxis]
        self.freq_range = fit_range

        self.amplitude = self.amplitude_spectrum()

        self.p0 = [1, -1e-3, -1e-4]
        self.bounds = (-np.inf, [np.inf, 1e-2, 1e-3])
        self.sigma = np.maximum(self.amplitude * 0.1,
                                np.maximum(d, 1e-18))

    def set_f0(self, f0):
        self.f0 = f0
        self.fit_fun = lambda f, a_0, b, c: a_0 * np.exp(b * (f - f0) + c * (f - f0) ** 2) + self.d

    def amplitude_spectrum(self):
        """
        Calculate the amplitude spectrum using NumPy's default normalisation. This is not the physically relevant
        spectrum, but it ensures consistency when applying the IFT later.
        """
        spectrum = np.apply_along_axis(np.fft.rfft, 0, self.time_trace)
        amplitude = np.apply_along_axis(np.abs, 0, spectrum)

        return amplitude

    def fit_amplitude(self):
        """
        Fit the amplitude spectrum in the range defined by self.freq_range, using the function A * exp( b * (f-f0) +
        c * (f-f0)**2 ).
        """
        # Set p0 with A[f0] ?
        freq = np.fft.rfftfreq(len(self.time_trace), 2e-10)
        freq_ind = np.logical_and(self.freq_range[0] <= freq,
                                  self.freq_range[1] >= freq)

        self.sigma = self.sigma[freq_ind]
        super().fit(freq[freq_ind] / 1e6, self.amplitude[freq_ind])

    def fit_amplitude_iter(self):
        """
        Fit the amplitude in an iterative fashion.
        """
        freq = np.fft.rfftfreq(len(self.time_trace), 2e-10)
        freq_ind = np.logical_and(self.freq_range[0] <= freq,
                                  self.freq_range[1] >= freq)

        self.sigma = self.sigma[freq_ind]

        self.fit_fun = lambda f, a, b: a * np.exp(b * (f - self.f0))
        self.p0 = [0.01, 1e-4]
        self.bounds = (-np.inf, [np.inf, 1e2])
        super().fit(freq[freq_ind] / 1e6, self.amplitude[freq_ind])

        self.fit_fun = [lambda f, b, c: a * np.exp(b * (f - self.f0) + c * (f - self.f0) ** 2) for a in self.opt[0, :]]
        self.p0 = [[b, -2e-6] for b in self.opt[1, :]]
        self.bounds = [(-np.inf, [1e2, 1e3]) for _ in range(3)]
        super().fit_multi(freq[freq_ind] / 1e6, self.amplitude[freq_ind])


class ParameterFitter(Fitter):
    """
    This object fits parameters with a quadratic function, using the first dimension as the observations and second
    dimension as the list of parameters. Each parameter is fitted using the same list of x-values.
    """
    def __init__(self, params, std_params, x_values):
        super().__init__(lambda x, p0, p1, p2: p0 + p1 * x + p2 * x ** 2)
        self.p0 = [1, 1, 1]
        self.sigma = std_params

        self.x = x_values
        self.parameters = params

    def fit_parameters(self):
        opt = np.zeros((3, 3, self.parameters.shape[2]))
        cov = np.zeros((3, 3, self.parameters.shape[2]))
        for pol in range(self.parameters.shape[2]):
            super().fit(self.x, self.parameters[:, :, pol])
            opt[:, :, pol] = self.opt
            cov[:, :, pol] = self.cov
        self.opt = opt
        self.cov = cov

    def save_to_file(self, file):
        write_file_json(file, self.opt)

    def load_from_file(self, file):
        self.opt = read_file_json(file)
