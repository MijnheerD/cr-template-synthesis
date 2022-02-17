import os
import sys
import glob
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.optimize import fmin
from GlobalVars import FIG_DIRECTORY, REAS_DIRECTORY, DISTANCES
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

            sims_part[nr] = (n_charged[slice_pos], n_charged[slice_pos] > 1e-6 * max(n_charged))

    return sims_part


def read_fitfile(file, x_slice, antenna):
    ar = np.genfromtxt(file, delimiter=', ', dtype=[('slice', int), ('antenna', int),
                                                    ('A', np.float64), ('b', np.float64), ('c', np.float64),
                                                    ('Xmax', np.float32), ('E', np.float32)])

    return ar[ar['slice'] == x_slice][antenna]


def add_par_to_fig(axis, x, y, mask, color, colormap, normalisation):
    axis.scatter(x[mask], y[mask], c=color[mask], cmap=colormap, norm=normalisation)
    axis.scatter(x[~mask], y[~mask], c=color[~mask], cmap=colormap, norm=normalisation, marker='x')


def add_target_to_fig(axis, x_vals, y_vals):
    for j in range(len(axis)):
        axis[j].scatter(x_vals[j], y_vals[j], marker='o', facecolors='none', edgecolors='r')


# General parameters
FIG_DIRECTORY = os.path.join(FIG_DIRECTORY, 'AmplitudeFitRL')
TARGET = '100000'

if __name__ == "__main__":
    if len(sys.argv) > 1:
        FIT_DIRECTORIES = [f'fitFiles_{sys.argv[1]}_17', f'fitFiles_{sys.argv[1]}_18', f'fitFiles_{sys.argv[1]}_19']
        X_SLICE = int(sys.argv[2])
        ANTENNA = int(sys.argv[3])
        # FIT_DIRECTORIES = [os.path.join('Amplitude spectrum fitting', entry) for entry in FIT_DIRECTORIES]
    else:
        FIT_DIRECTORIES = ['fitFilesUB17', 'fitFilesUB18', 'fitFilesUB19']
        X_SLICE = 600
        ANTENNA = 3

# Make the figures to plot the parameters
plt.style.use('dark_background')
fig1, [ax1, ax2] = plt.subplots(1, 2, figsize=(15, 6))
fig2, [ax3, ax4] = plt.subplots(1, 2, figsize=(15, 6))
fig3, [ax5, ax6] = plt.subplots(1, 2, figsize=(15, 6))

axX = [ax1, ax3, ax5]
axY = [ax2, ax4, ax6]

cmap = cm.get_cmap('plasma')
norm = colors.Normalize(vmin=190, vmax=290, clip=True)

# Construct the required dictionaries and file handlers
simsRL = sims_r_l(REAS_DIRECTORY)
simsPART = sims_particles(REAS_DIRECTORY, X_SLICE)

target_x = None
target_y = None
fitFiles = ReaderWriter()
for entry in FIT_DIRECTORIES:
    fitFiles.open_all_files(entry)

    parametersX = []
    parametersY = []
    for ind, (fileX, fileY) in enumerate(zip(fitFiles.filesX, fitFiles.filesY)):
        name = os.path.basename(os.path.normpath(fileX.name)).split('.')[0][3:]
        proton = True if name.startswith('1') else False

        if simsPART[name][1]:
            R, L, X_max = simsRL[name]

            lineX = read_fitfile(fileX, X_SLICE, ANTENNA)
            lineY = read_fitfile(fileY, X_SLICE, ANTENNA)

            parametersX.append([X_max, lineX['A'] / simsPART[name][0], lineX['b'], lineX['c'], L, proton])
            parametersY.append([X_max, lineY['A'] / simsPART[name][0], lineY['b'], lineY['c'], L, proton])

            if name == TARGET:
                target_x = [X_max] * 6
                target_y = [lineX['A'] / simsPART[name][0], lineX['b'], lineX['c'],
                            lineY['A'] / simsPART[name][0], lineY['b'], lineY['c']]

    arX = np.array(parametersX)
    arY = np.array(parametersY)

    X = arX[:, 0]
    L = arX[:, 4]

    for i, (x_ax, y_ax) in enumerate(zip(axX, axY)):
        add_par_to_fig(x_ax, X, arX[:, i + 1], arX[:, 5].astype(bool), L, cmap, norm)
        add_par_to_fig(y_ax, X, arY[:, i + 1], arX[:, 5].astype(bool), L, cmap, norm)

    fitFiles.close_all_files()

# Add target marker
if target_x is not None:
    add_target_to_fig(axX + axY, target_x, target_y)

# Add labels and colorbar to figures
y_labels = [r'$A_0 / N_{slice}$ [$V/ \mu m$]', r'b [$1 / MHz$]', r'c [$1 / MHz^2$]']

for i, (x_ax, y_ax) in enumerate(zip(axX, axY)):
    x_ax.set_xlabel(r'$X_{max}$ [$g/cm^2$]')
    x_ax.set_ylabel(y_labels[i] + '(x-component)')
    x_ax.set_title(f'X = {int(X_SLICE)} g/cm^2, r = {int(DISTANCES[ANTENNA] / 100)} m')

    y_ax.set_xlabel(r'$X_{max}$ [$g/cm^2$]')
    y_ax.set_ylabel(y_labels[i] + '(y-component)')
    y_ax.set_title(f'X = {int(X_SLICE)} g/cm^2, r = {int(DISTANCES[ANTENNA] / 100)} m')

fig1.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=[ax1, ax2], location='right', shrink=1)
fig2.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=[ax3, ax4], location='right', shrink=1)
fig3.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=[ax5, ax6], location='right', shrink=1)

# Go to figure directory and save all figures
os.chdir(FIG_DIRECTORY)
fig1.savefig(f'A0_L_{ANTENNA}x{X_SLICE}.png', bbox_inches='tight')
fig2.savefig(f'b_L_{ANTENNA}x{X_SLICE}.png', bbox_inches='tight')
fig3.savefig(f'c_L_{ANTENNA}x{X_SLICE}.png', bbox_inches='tight')
