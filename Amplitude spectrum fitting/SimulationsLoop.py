import os
import glob
import numpy as np
import concurrent.futures as futures
from scipy.optimize import curve_fit
from scipy.constants import c as c_vacuum


DISTANCES = [1, 4000, 7500, 11000, 15000, 37500]  # antenna radial distances to shower core, in cm
WORKING_PATH = "fitFiles7/"  # directory to store all the files containing the fit parameters
LOG_FILE = "amp_fits.log"  # file to log use of linear fit


def amplitude_fit_slice(fdata, ydata, x, r):
    d = max(0, 1e-9 * (x/400 - 1.5) * np.exp(1 - r/40000))
    try:
        popt, pcov = curve_fit(lambda f, a_0, b, c: a_0 * np.exp(b * f + c * f ** 2), fdata/1e6, ydata - d**2,
                               p0=[25, -1e-3, -1e-4], bounds=(-np.inf, [np.inf, 1e-2, 1e-3]))
        linear = False
    except RuntimeError:
        popt, pcov = curve_fit(lambda f, a_0, b, c: a_0 + b * f + c * f**2, fdata/1e6, np.log(ydata - d**2),
                               p0=[np.log(25), -1e-3, -1e-6])
        popt[0] = np.exp(popt[0])  # The first variable in the linear fit is ln(A_0)
        linear = True
    return popt, d**2, linear


def analyze_file(path_to_file, filename):
    """
    Analyses a single CoREAS simulations file, located at path_to_file from the current working directory and with name
    filename, which should be of the form 'raw_{ANTENNA}x{SLICE}.dat'
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
        coefX, dX, linear = amplitude_fit_slice(freq[frange], filtered[:, 0], Xslice, DISTANCES[antenna])
        if linear:
            with open(LOG_FILE, 'a+') as f:
                f.write(os.path.join(path_to_file, filename))
                f.write(" used linear fit for X\n")
        coefY, dY, linear = amplitude_fit_slice(freq[frange], filtered[:, 1], Xslice, DISTANCES[antenna])
        if linear:
            with open(LOG_FILE, 'a+') as f:
                f.write(os.path.basename(os.getcwd()))
                f.write(filename)
                f.write("used linear fit for Y\n")

    return [[Xslice, antenna, *coefX], [Xslice, antenna, *coefY]]


def analyze_directory_threads(path):
    # os.chdir(path)  # Don't change dir, otherwise next iteration will be working in wrong dir
    with futures.ThreadPoolExecutor(max_workers=10) as executor:
        res = executor.map(analyze_file, [path]*len(os.listdir(path)), os.listdir(path), chunksize=10)
    return list(res)


def analyze_directory(path):
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
        res.append(analyze_file(path, file))
    return res


def analyze_simulation(dir_path):
    """
    Analyze the simulation contained in dir_path, which should be of the form "SIMxxxxxx_coreas/" . This function
    assumes that the current working directory contains the .reas file of the simulation, as well as the directory
    containing the CoREAS files.
     """
    sim_number = dir_path.split('/')[-2].split('_')[0]
    # working_directory = os.getcwd()

    with open(sim_number + '.reas', 'r') as file:
        for line in file:
            if 'DepthOfShowerMaximum' in line:
                Xmax = float(line.split()[2])
            elif 'PrimaryParticleEnergy' in line:
                primaryE = float(line.split()[2])

    results = np.array(analyze_directory(dir_path))
    sim_var = np.array([[Xmax, primaryE]]*len(results))
    resX = np.hstack((results[:, 0, :], sim_var))
    resY = np.hstack((results[:, 1, :], sim_var))

    with open(os.path.join(WORKING_PATH, 'fitX', f'{sim_number}.dat'), 'w') as file:
        np.savetxt(file, resX, fmt='%.1d, %.1d, %.13e, %.13e, %.13e, %.5g, %.1e')
    with open(os.path.join(WORKING_PATH, 'fitY', f'{sim_number}.dat'), 'w') as file:
        np.savetxt(file, resY, fmt='%.1d, %.1d, %.13e, %.13e, %.13e, %.5g, %.1e')

    print(f'Done with {sim_number}...')

    return sim_number, len(results)


def fit_simulations(path):
    """
    Fit all the simulations in the directory 'path' using multiprocessing.
    :param path: Directory containing all the simulations
    :return: List of tuples, each containing the name and length of the time trace of a simulation.
    """
    from datetime import date

    global WORKING_PATH, LOG_FILE
    WORKING_PATH = os.path.join(os.getcwd(), WORKING_PATH)
    LOG_FILE = os.path.join(os.getcwd(), LOG_FILE)
    os.chdir(path)

    with open(LOG_FILE, 'a+') as file:
        file.write(str(date.today()))

    with futures.ProcessPoolExecutor() as executor:
        res = executor.map(analyze_simulation, glob.glob('SIM*/'))
    return list(res)


directory = '/mnt/hgfs/Shared data/BulkSynth/bulksynth-7/'
lengths = fit_simulations(directory)
print(lengths)
