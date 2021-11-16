import os
import numpy as np
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
from crsynthesizer.FileReaderWriter import ReaderWriter


def get_number_of_particles(path):
    long = np.genfromtxt(path, skip_header=2, skip_footer=216, usecols=(2, 3))
    return np.sum(long, axis=1)


def make_profile(bins, x_data, y_data):
    mean = binned_statistic(x_data, y_data, statistic='mean', bins=len(bins) - 1, range=(min(bins), max(bins)))
    std = binned_statistic(x_data, y_data, statistic='std', bins=len(bins) - 1, range=(min(bins), max(bins)))
    return mean[0], std[0]


def fit_profile(fitfun, bins, mean, std):
    bin_centers = (bins[1:] + bins[:-1]) / 2
    to_fit = list(zip(*[(bin_centers[i], mean[i], std[i]) for i in range(len(bin_centers)) if std[i] != 0]))
    popt, pcov = curve_fit(fitfun, to_fit[0], to_fit[1], sigma=to_fit[2])
    return popt, pcov


FIT_DIRECTORIES = ['fitFiles5017', 'fitFiles5018', 'fitFiles5019']
COLORS = ['cyan', 'magenta', 'yellow']
REAS_DIRECTORY = 'CORSIKA_long_files'
PARAM_DIRECTORY = 'paramProfileFit50'
DISTANCES = [1, 4000, 7500, 11000, 15000, 37500]

# Open all simulation files, grouped per polarization
handler = ReaderWriter()
for directory in FIT_DIRECTORIES:
    handler.open_all_files(directory)

# Construct a list of all X_max values and the amplitude fit values per slice/antenna/file
X_max_tot = []
A_x = np.zeros((207, 6, len(handler.filesX)))
A_y = np.zeros((207, 6, len(handler.filesX)))
b_x = np.zeros((207, 6, len(handler.filesX)))
b_y = np.zeros((207, 6, len(handler.filesX)))
c_x = np.zeros((207, 6, len(handler.filesX)))
c_y = np.zeros((207, 6, len(handler.filesX)))

for ind, (fileX, fileY) in enumerate(zip(handler.filesX, handler.filesY)):
    name = os.path.basename(os.path.normpath(fileX.name)).split('.')[0]

    # Read in the fit files and sort by slice
    arX = np.genfromtxt(fileX, delimiter=', ', dtype=[('slice', np.float32), ('antenna', int),
                                                      ('A', np.float64), ('b', np.float64), ('c', np.float64),
                                                      ('Xmax', np.float32), ('E', np.float32)])
    arX.sort(axis=0, order=['antenna', 'slice'])

    arY = np.genfromtxt(fileY, delimiter=', ', dtype=[('slice', np.float32), ('antenna', int),
                                                      ('A', np.float64), ('b', np.float64), ('c', np.float64),
                                                      ('Xmax', np.float32), ('E', np.float32)])
    arY.sort(axis=0, order=['antenna', 'slice'])

    particle_numbers = get_number_of_particles(os.path.join('CORSIKA_long_files', 'DAT' + name[3:] + '.long'))
    particle_numbers[np.where(particle_numbers==0)] = 1

    nr_of_slices = len(particle_numbers)
    nr_of_antennas = int(len(arX) / nr_of_slices)

    # Append necessary values
    X_max_tot.append(arX['Xmax'][0])

    for antenna in range(nr_of_antennas):
        sliceX = arX[int(antenna*nr_of_slices):int((antenna+1)*nr_of_slices)]
        sliceY = arY[int(antenna * nr_of_slices):int((antenna + 1) * nr_of_slices)]

        A_x[:, antenna, ind] = sliceX['A'] / particle_numbers
        A_y[:, antenna, ind] = sliceY['A'] / particle_numbers

        b_x[:, antenna, ind] = sliceX['b']
        b_y[:, antenna, ind] = sliceY['b']

        c_x[:, antenna, ind] = sliceX['c']
        c_y[:, antenna, ind] = sliceY['c']


