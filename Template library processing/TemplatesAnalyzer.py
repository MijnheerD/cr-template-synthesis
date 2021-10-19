import os
import glob
import numpy as np
import concurrent.futures as futures
from scipy.optimize import curve_fit
from scipy.constants import c as c_vacuum


def amplitude_fit_slice(fdata, ydata, x, r, f0=0):
    d = max(0, 1e-9 * (x / 400 - 1.5) * np.exp(1 - r / 40000))
    try:
        popt, pcov = curve_fit(lambda f, a_0, b, c: a_0 * np.exp(b * (f - f0) + c * (f - f0) ** 2), fdata / 1e6,
                               ydata - d ** 2, p0=[25, -1e-3, -1e-4], bounds=(-np.inf, [np.inf, 1e-2, 1e-3]),
                               sigma=np.maximum(ydata*0.1, np.maximum(d**2, 1e-18)))
        linear = False
    except RuntimeError:
        popt, pcov = curve_fit(lambda f, a_0, b, c: a_0 + b * (f - f0) + c * (f - f0) ** 2, fdata / 1e6,
                               np.log(ydata - d ** 2), p0=[np.log(25), -1e-3, -1e-6],
                               sigma=np.log(np.maximum(ydata*0.1, np.maximum(d**2, 1e-18))))
        popt[0] = np.exp(popt[0])  # The first variable in the linear fit is ln(A_0)
        linear = True
    return popt, d ** 2, linear


def fit_parameters(fitfun, x_data: list, y_data: list, return_cov=False):
    """
    Fits multiple parameters to the same fit function. Pass all the data for each parameter as an entry in y_data.
    :param lambda fitfun: Function to fit to the data.
    :param x_data: Data on the x-axis to be used for every fit.
    :param y_data: Data on the y-axis, can be a single list or a list of data sets.
    :param return_cov: Whether to return the covariance matrix together with the fitted parameters. If this is True,
    the output will be a list of tuples each having the best fit parameters as their first element and the corresponding
    covariance matrix as second elements.
    :return: List of same length as y_data, each element being the set of best fit parameters.
    """
    fits = []

    # If y_data does not containing multiple parameters to fit
    if not (isinstance(y_data[0], list) or isinstance(y_data[0], np.ndarray)):
        y_data = [y_data]

    # Fit every parameter present in y_data
    for y in y_data:
        res, cov = curve_fit(fitfun, x_data, y)
        fits.append(res) if not return_cov else fits.append((res, cov))

    return fits


def open_all_files(path):
    """
    Open all files in a directory.
    :param str path: Path to the directory
    :return list: List of all the file handles of the opened files.
    """
    files = []
    for file in os.listdir(path):
        files.append(open(os.path.join(path, file)))
    return files


def close_all_files(handle_list):
    """
    Close all the files in the list.
    :param list handle_list: List of file handles, to be closed.
    """
    for file in handle_list:
        file.close()


def get_number_of_particles(path):
    """
    Get the sum of electrons and positrons per slice, for every file in path. If path is a string, it is considered
    as a single file to analyze (ie a list with 1 entry).
    :param path: Path to file or list of paths from where to extract the particle counts.
    :type path: str or list
    :return np.ndarray: Particles counts, formatted as np.ndarray with dimensions (len(path), number_of_slices).
    """
    if type(path) == str:
        long = np.genfromtxt(path, skip_header=2, skip_footer=216, usecols=(2, 3))
        return np.sum(long, axis=1)
    elif type(path) == list:
        res = np.array([np.sum(np.genfromtxt(file, skip_header=2, skip_footer=216, usecols=(2, 3)), axis=1)
                        for file in path])
        return res


class TemplatesAnalyzer(object):
    """
    Object to analyze all the CoREAS simulations in a directory, to extract the parameters relevant to cosmic ray
    template synthesis.
    :param template_directory: Path to the directory containing the simulations (.reas files and directories).
    :param working_path: Path to the directory where the fit parameters should be stored, relative to template_directory
    :param distances: List of the antenna radial distances to the shower core, ordered as the CoREAS numbering.
    :param log_file: Path to file to log fit errors, relative to template_directory.
    :param parameter_path: Path to the directory where to store the quadratic fit parameters, relative to the
    template_directory.
    """

    def __init__(self, template_directory, working_path, distances, log_file="amp_fits.log", **kwargs):
        self.directory = template_directory
        self.working_path = working_path
        self.param_path = kwargs.get('parameter_path', working_path)
        self.distances = distances
        self.log_file = os.path.join(os.getcwd(), log_file)
        self.fit_center = 0

        self._check_dirs()

    def _check_dirs(self):
        """
        Check whether the necessary directories exist, otherwise create them.
        """
        if not os.path.exists(self.working_path):
            os.mkdir(self.working_path)
            os.mkdir(os.path.join(self.working_path, "fitX/"))
            os.mkdir(os.path.join(self.working_path, "fitY/"))

        if not os.path.exists(self.param_path):
            os.mkdir(self.param_path)
            os.mkdir(os.path.join(self.param_path, "fitX/"))
            os.mkdir(os.path.join(self.param_path, "fitY/"))

    def _analyze_file(self, path_to_file: str, filename: str):
        """
        Analyses a single CoREAS simulations file, located at path_to_file from the current working directory and with
        filename, which should be of the form 'raw_{ANTENNA}x{SLICE}.dat'
        :param path_to_file: Path to the directory containing the file, relative to the directory of the
        TemplatesAnalyzer instance.
        :param filename: Name of the file to analyze.
        :return: Lists containing the X [g/cm^2] of the slice and the number of the antenna, together with the
        amplitude best fit parameters, for the x and y polarization polarization.
        """
        # Find the antenna number and atm. depth of slice
        temp = filename.split('_')[1].split('.')[0]
        antenna, Xslice = map(int, temp.split('x'))

        with open(os.path.join(path_to_file, filename), 'r') as file:
            # Read in data from .dat simulation file and convert to microV/m
            data = np.genfromtxt(file) * c_vacuum * 1e2

            # Calculate frequency values and filter range
            freq = np.fft.rfftfreq(len(data), 2e-10)
            frange = np.logical_and(10 * 1e6 <= freq, freq <= 5 * 1e8)

            # Calculate the amplitude spectrum and band-pass filter it
            spectrum = np.apply_along_axis(np.fft.rfft, 0, data[:, 1:])
            amplitude = np.apply_along_axis(np.abs, 0, spectrum)
            filtered = amplitude[frange]

            # Fit the amplitude spectrum, report if linear fit is used
            coefX, dX, linear = amplitude_fit_slice(freq[frange], filtered[:, 0], Xslice, self.distances[antenna],
                                                    f0=self.fit_center)
            if linear:
                with open(self.log_file, 'a+') as f:
                    f.write(os.path.join(path_to_file, filename))
                    f.write(" used linear fit for X\n")
            coefY, dY, linear = amplitude_fit_slice(freq[frange], filtered[:, 1], Xslice, self.distances[antenna],
                                                    f0=self.fit_center)
            if linear:
                with open(self.log_file, 'a+') as f:
                    f.write(os.path.join(path_to_file, filename))
                    f.write(" used linear fit for Y\n")

        return [[Xslice, antenna, *coefX], [Xslice, antenna, *coefY]]

    def _analyze_directory(self, path: str):
        """
        Analyze the directory containing all the CoREAS from 1 simulation. The directory must not be contained in the
        current working directory, as long as path is set correctly.
        :param path: Path from current working directory to directory to analyze.
        :return: List of lists, each containing a list of the form [X_slice, antenna_number, fit_values] for each
        polarization.
        """
        res = []
        print(f'Analyzing {path}...')
        for file in os.listdir(path):
            res.append(self._analyze_file(path, file))
        return res

    def _analyze_simulation(self, dir_path: str):
        """
        Analyze the simulation contained in dir_path, which should be of the form "SIMxxxxxx_coreas/" . This function
        assumes that the current working directory contains the .reas file of the simulation, as well as the directory
        containing the CoREAS files.
         """
        sim_number = dir_path.split('/')[-2].split('_')[0]

        with open(sim_number + '.reas', 'r') as file:
            for line in file:
                if 'DepthOfShowerMaximum' in line:
                    Xmax = float(line.split()[2])
                elif 'PrimaryParticleEnergy' in line:
                    primaryE = float(line.split()[2])

        results = np.array(self._analyze_directory(dir_path))
        sim_var = np.array([[Xmax, primaryE]] * len(results))
        resX = np.hstack((results[:, 0, :], sim_var))
        resY = np.hstack((results[:, 1, :], sim_var))

        with open(os.path.join(self.working_path, 'fitX', f'{sim_number}.dat'), 'w') as file:
            np.savetxt(file, resX, fmt='%.1d, %.1d, %.13e, %.13e, %.13e, %.5g, %.1e')
        with open(os.path.join(self.working_path, 'fitY', f'{sim_number}.dat'), 'w') as file:
            np.savetxt(file, resY, fmt='%.1d, %.1d, %.13e, %.13e, %.13e, %.5g, %.1e')

        print(f'Done with {sim_number}...')

        return sim_number, len(results)

    def _save_antenna_to_slice_file(self, antenna, data, path):
        with open(os.path.join(self.param_path, path), 'a+') as f:
            f.write('antenna' + antenna + '\t')
            for res in data:
                f.writelines(map(lambda x: str(x) + '\t', res))
            f.write('\n')

    def fit_parameters(self):
        """
        Perform a quadratic fit to the amplitude parameters extracted by _fit_simulation. If the files with the fitted
        parameters are split over multiple directories, set self.working_path to a list containing all the directories
        to be used before calling this function. Use self.directory to point to the directories containing the .reas
        files in that case.
        """
        x_max = []  # Xmax values present in all the simulations
        x_max_filled = False  # Read Xmax data only once, as it stays the same for every iteration

        # Convert paths to lists, to be consistent in creating file handle lists
        if type(self.working_path) == str:
            working_path = [self.working_path]
            reas_path = [self.directory]
        else:
            working_path = self.working_path
            reas_path = self.directory

        # List of file handles of amplitude fit files for every simulation
        files_x = []
        files_y = []
        for path in working_path:
            files_x.extend(open_all_files(os.path.join(path, 'fitX/')))
            files_y.extend(open_all_files(os.path.join(path, 'fitY/')))

        # Find number of particles in each slice for every simulation
        long_files = [os.path.join(reas_path[working_path.index(file.name.split('fitX')[0])],
                                   'DAT' + file.name.split('/')[-1].split('.')[0].split('SIM')[1] + '.long')
                      for file in files_x]
        n_particles = get_number_of_particles(long_files)

        # Loop over all slices, aka every line in every file
        while True:
            # Reset parameter lists for X and Y
            a_0_x, b_x, c_x = [], [], []
            a_0_y, b_y, c_y = [], [], []

            # Loop over all the files in polarization pairs
            for (fileX, fileY) in zip(files_x, files_y):
                # Read one line of every file and split it into a list
                lst_x = fileX.readline().split(', ')
                lst_y = fileY.readline().split(', ')

                # Check if end of file is reached
                if len(lst_x) == 1:
                    # Close all the files once done
                    close_all_files(files_x)
                    close_all_files(files_y)
                    return

                if not x_max_filled:
                    # Check if the Xmax of X and Y is the same (should not be necessary)
                    assert lst_x[5] == lst_y[5], f"X and Y have different Xmax, {lst_x[5]} and {lst_y[5]}"
                    x_max.append(float(lst_x[5]))

                # Collect the parameter values per slice, per antenna
                a_0_x.append(float(lst_x[2])), b_x.append(float(lst_x[3])), c_x.append(float(lst_x[4]))
                a_0_y.append(float(lst_y[2])), b_y.append(float(lst_y[3])), c_y.append(float(lst_y[4]))

            # Get the number of particles in the current slice for every simulation
            n_slice = n_particles[:, int(int(lst_x[0]) / 5 - 1)]
            n_slice[np.where(n_slice == 0)] = 1  # Avoid division by 0

            # Perform a quadratic fit to every parameter (for current combination of slice and antenna)
            a_0_x, a_0_y = np.array(a_0_x), np.array(a_0_y)
            res_x_a, res_x_b, res_x_c, res_y_a, res_y_b, res_y_c = \
                fit_parameters(lambda x, p0, p1, p2: p0 + p1 * x + p2 * x ** 2, x_max,
                               [a_0_x / n_slice, b_x, c_x, a_0_y / n_slice, b_y, c_y])

            # Save fitted parameters to file corresponding to current slice
            self._save_antenna_to_slice_file(lst_x[1], [res_x_a, res_x_b, res_x_c],
                                             os.path.join('fitX', 'slice' + lst_x[0] + '.dat'))
            self._save_antenna_to_slice_file(lst_y[1], [res_y_a, res_y_b, res_y_c],
                                             os.path.join('fitY', 'slice' + lst_y[0] + '.dat'))

            x_max_filled = True

    def fit_simulations(self):
        """
        Fit all the simulations in the directory 'path' using multiprocessing.
        """
        from datetime import date

        os.chdir(self.directory)

        with open(self.log_file, 'a+') as file:
            file.write(str(date.today()) + '\n')

        with futures.ProcessPoolExecutor() as executor:
            executor.map(self._analyze_simulation, glob.glob('SIM*/'))

    def fit_templates(self, center=0):
        """
        Fit the templates provided in the template_directory. First the amplitude spectrum in every antenna is
        calculated per slice (defined in the CoREAS files) and this is fitted to an exponential function. The best fit
        parameters are stored in the working_path. Subsequently these best fit parameters are fitted against X_max with
        a quadratic function. These best fit parameters are stored per slice in the parameter_path, with each antenna
        corresponding to 1 line.
        :param float center: Sets the f0 parameter in the fit function A * exp(b * [f - f0] + c * [f - f0]^2)
        """
        # Set the f0 parameter in the fit function
        self.fit_center = center

        # Fit the amplitude spectrum per slice and per antenna for every simulation in self.directory
        self.fit_simulations()

        # Fit the amplitude parameters with a quadratic function
        self.fit_parameters()
