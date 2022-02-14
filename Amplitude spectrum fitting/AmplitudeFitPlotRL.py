import os
import glob
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.optimize import fmin
from crsynthesizer.FileReaderWriter import working_directory, ReaderWriter


def shape_r_l(p, x):
    x1 = x - p[2]
    fn = np.exp(-x1 / (p[0] * p[1])) * np.power((1 + p[0] * x1 / p[1]), 1 / (p[0] * p[0]))

    return fn


def residual_mse(p, x, y):
    res = np.sum((y - shape_r_l(p, x)) ** 2)

    return res


def fit_r_l(atm_depth, n_tot):  # longitudinal profile :slanted depth vs # particle(em/charged)
    n1 = n_tot / np.max(n_tot)
    int_rl = np.array([0.3, 210, 620])
    parfit = fmin(residual_mse, int_rl.tolist(), args=(np.array(atm_depth), n1), disp=False)

    return parfit


def read_long(file):
    long = np.genfromtxt(file, skip_header=2, skip_footer=216, usecols=(0, 2, 3))
    return long[:, 0], np.sum(long[:, 1:], axis=1)


def sims_r_l(directory):
    """
    Construct a dictionary with the simulation numbers as keys and a tuple of R, L and X_max as value.
    :param directory: Directory containing the .long files of the simulations.
    :return: The dictionary with R, L and X_max per simulation.
    """
    sims_dict = {}

    with working_directory(directory):
        long_files = glob.glob('DAT*.long')

        for file in long_files:
            nr = file.split('.')[0][3:]
            x_slices, n_charged = read_long(file)

            sims_dict[nr] = fit_r_l(x_slices[10:-10], n_charged[10:-10])  # R, L, X_max

    return sims_dict


def sims_particles(directory, x_slice):
    sims_part = {}

    with working_directory(directory):
        long_files = glob.glob('DAT*.long')

        for file in long_files:
            nr = file.split('.')[0][3:]
            x_slices, n_charged = read_long(file)

            slice_pos = np.where(x_slices.astype('int') == int(x_slice))[0][0]

            sims_part[nr] = n_charged[slice_pos] < 1e-6 * max(n_charged)

    return sims_part


def read_fitfile(file, x_slice, antenna):
    ar = np.genfromtxt(file, delimiter=', ', dtype=[('slice', int), ('antenna', int),
                                                    ('A', np.float64), ('b', np.float64), ('c', np.float64),
                                                    ('Xmax', np.float32), ('E', np.float32)])

    return ar[ar['slice'] == x_slice][antenna]


def add_par_to_fig(axis, x, y, mask, color, colormap, normalisation):
    axis.scatter(x[mask], y[mask], c=color[mask], cmap=colormap, norm=normalisation)
    axis.scatter(x[~mask], y[~mask], c=color[~mask], cmap=colormap, norm=normalisation, marker='x')


# General parameters
REAS_DIRECTORY = '/mnt/hgfs/Shared data/BulkSynth/CORSIKA_long_files'
FIT_DIRECTORIES = ['fitFilesUB17', 'fitFilesUB18', 'fitFilesUB19']

X_SLICE = 600
ANTENNA = 3

# Construct the required dictionaries and file handlers
simsRL = sims_r_l(REAS_DIRECTORY)
simsPART = sims_particles(REAS_DIRECTORY, X_SLICE)

fitFiles = ReaderWriter()
for entry in FIT_DIRECTORIES:
    fitFiles.open_all_files(entry)

parametersX = []
parametersY = []
for ind, (fileX, fileY) in enumerate(zip(fitFiles.filesX, fitFiles.filesY)):
    name = os.path.basename(os.path.normpath(fileX.name)).split('.')[0][3:]
    proton = True if name.startswith('1') else False

    if simsPART[name]:
        R, L, X_max = simsRL[name]

        lineX = read_fitfile(fileX, X_SLICE, ANTENNA)
        lineY = read_fitfile(fileY, X_SLICE, ANTENNA)

        parametersX.append((X_max, lineX['A'], lineX['b'], lineX['c'], L, proton))
        parametersX.append((X_max, lineY['A'], lineY['b'], lineY['c'], L, proton))

arX = np.array(parametersX)
arY = np.array(parametersY)

# Make the figures to plot the parameters
plt.style.use('dark_background')
fig1, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 6))
fig2, [ax3, ax4] = plt.subplots(1, 2, figsize=(12, 6))
fig3, [ax5, ax6] = plt.subplots(1, 2, figsize=(12, 6))

axX = [ax1, ax3, ax5]
axY = [ax2, ax4, ax6]

X = arX[:, 0]
L = arX[:, 4]

cmap = cm.get_cmap('plasma')
norm = colors.Normalize(vmin=min(L), vmax=max(L))

for i, (x_ax, y_ax) in enumerate(zip(arX, arY)):
    add_par_to_fig(x_ax, X, arX[:, i + 1], arX[:, 5], L, cmap, norm)
    add_par_to_fig(y_ax, X, arY[:, i + 1], arX[:, 5], L, cmap, norm)

fig1.savefig(f'A0_L_{ANTENNA}x{X_SLICE}.png', bbox_inches='tight')
fig2.savefig(f'b_L_{ANTENNA}x{X_SLICE}.png', bbox_inches='tight')
fig3.savefig(f'c_L_{ANTENNA}x{X_SLICE}.png', bbox_inches='tight')
