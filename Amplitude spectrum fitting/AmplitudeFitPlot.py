import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
from scipy.constants import c


def get_number_of_particles(path):
    long = np.genfromtxt(path, skip_header=2, skip_footer=216, usecols=(2, 3))
    return np.sum(long, axis=1)


def make_profile(bins, x_data, y_data):
    mean = binned_statistic(x_data, y_data, statistic='mean', bins=bins)
    std = binned_statistic(x_data, y_data, statistic='std', bins=bins)
    return mean[0], std[0]


def fit_profile(fitfun, bins, mean, std):
    bin_centers = (bins[1:] + bins[:-1]) / 2
    to_fit = list(zip(*[(bin_centers[i], mean[i], std[i]) for i in range(len(bin_centers)) if std[i] != 0]))
    popt, pcov = curve_fit(fitfun,  to_fit[0], to_fit[1],
                           p0=[np.median(to_fit[1]), 1e-6, -1e-10],
                           sigma=np.clip(to_fit[2], np.percentile(to_fit[2], 3), None))
    return popt, pcov


def plot_fit_profile(ax, x_plot, param, bins, mean, std, bar=False):
    width = int(bins[1] - bins[0])
    ax.plot(x_plot, param[0] + param[1] * x_plot + param[2] * x_plot ** 2, label='Fit to profile')
    if bar:
        ax.bar(bins[:-1], mean, align='edge', width=width, color='w', alpha=0.3, yerr=std, ecolor='w')


FIT_DIRECTORIES = ['fitFilesUB17', 'fitFilesUB18', 'fitFilesUB19']
FIT_TYPES = {'profile'}
PLOT_TYPES = {'fit', 'target'}
COLORS = ['cyan', 'magenta', 'yellow']
COLORS_DAVID = ['green', 'red', 'orange']
REAS_DIRECTORY = '/mnt/hgfs/Shared data/BulkSynth/CORSIKA_long_files'
DAVID_DIRECTORY = '/mnt/hgfs/Shared data/ampfitQ/'
PARAM_DIRECTORY = '../Parameters/paramProfileFitUB'
DISTANCES = [1, 4000, 7500, 11000, 15000, 37500]  # antenna_nr radial distances to shower core, in cm

# if __name__ == "__main__":
#     XSLICE = int(sys.argv[1])
#     ANTENNA = int(sys.argv[2])
XSLICE = 600
ANTENNA = 3
TARGET = '100000'

arX = np.genfromtxt(os.path.join(PARAM_DIRECTORY, 'fitX', 'slice' + str(XSLICE) + '.dat'))
arY = np.genfromtxt(os.path.join(PARAM_DIRECTORY, 'fitY', 'slice' + str(XSLICE) + '.dat'))

# Make the figures to plot the parameters
plt.style.use('dark_background')
fig1, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 6))
fig2, [ax3, ax4] = plt.subplots(1, 2, figsize=(12, 6))
fig3, [ax5, ax6] = plt.subplots(1, 2, figsize=(12, 6))

A_x_tot, A_y_tot = [], []
b_x_tot, b_y_tot = [], []
c_x_tot, c_y_tot = [], []
X_max_tot = []

iron_index = []
target_index = []
X_max_dict = {}

for ind, directory in enumerate(FIT_DIRECTORIES):
    A_0_x = []
    b_x = []
    c_x = []
    X_max_x = []

    A_0_y = []
    b_y = []
    c_y = []
    X_max_y = []

    proton = True
    iron_ind = 0
    for i, file in enumerate(os.listdir(os.path.join(directory, 'fitX'))):
        name = file.split('.')[0].split('SIM')[1]
        # Look if we are dealing with the first iron shower
        if name.startswith('2') and proton:
            proton = False
            iron_index.append(i)
        # Look for index in tot lists of target shower
        if name == TARGET:
            target_index.append(i + len(X_max_tot))
        # Check if shower has enough particles in this particular slice
        n_slice = get_number_of_particles(os.path.join(REAS_DIRECTORY,
                                                       'DAT' + file.split('.')[0].split('SIM')[1] + '.long'))
        if n_slice[int(XSLICE / 5 - 1)] < 1e-6 * max(n_slice):
            iron_ind -= 1
            continue

        with open(os.path.join(directory, 'fitX', file), 'r') as fX, \
                open(os.path.join(directory, 'fitY', file), 'r') as fY:
            for lineX, lineY in zip(fX, fY):
                lstX = lineX.split(', ')
                lstY = lineY.split(', ')
                if float(lstX[0]) == XSLICE:
                    if float(lstX[1]) == ANTENNA:
                        A_0_x.append(np.sqrt(float(lstX[2]) / n_slice[int(XSLICE / 5 - 1)] / c / 1e2))
                        b_x.append(float(lstX[3]))
                        c_x.append(float(lstX[4]))
                        X_max_x.append(float(lstX[5]))
                        A_0_y.append(np.sqrt(float(lstY[2]) / n_slice[int(XSLICE / 5 - 1)] / c / 1e2))
                        b_y.append(float(lstY[3]))
                        c_y.append(float(lstY[4]))
                        X_max_y.append(float(lstY[5]))
                        break  # We know there is only 1 interesting entry per file

        X_max_dict[name] = X_max_x[-1]

    iron_ind += iron_index[-1]
    ax1.scatter(X_max_x[:iron_ind], A_0_x[:iron_ind], color=COLORS[ind])
    ax1.scatter(X_max_x[iron_ind:], A_0_x[iron_ind:], color=COLORS[ind], marker='x')
    ax2.scatter(X_max_y[:iron_ind], A_0_y[:iron_ind], color=COLORS[ind])
    ax2.scatter(X_max_y[iron_ind:], A_0_y[iron_ind:], color=COLORS[ind], marker='x')

    ax1.set_xlabel(r"$X_{max}[g/cm^2]$")
    ax1.set_ylabel(r"$A_0 / N_{slice}$ (x-component) [$V / \mu m$]")
    ax1.set_xlim([500, 950])
    ax1.set_title(f"X = {XSLICE} g/cm^2 r = {DISTANCES[ANTENNA] / 100} m")
    ax1.ticklabel_format(axis='y', useMathText=True, scilimits=(0, 0))

    ax2.set_xlabel(r"$X_{max}[g/cm^2]$")
    ax2.set_ylabel(r"$A_0 / N_{slice}$ (y-component) [$V / \mu m$]")
    ax2.set_xlim([500, 950])
    ax2.set_title(f"X = {XSLICE} g/cm^2 r = {DISTANCES[ANTENNA] / 100} m")
    ax2.ticklabel_format(axis='y', useMathText=True, scilimits=(0, 0))

    ax3.scatter(X_max_x[:iron_ind], b_x[:iron_ind], color=COLORS[ind])
    ax3.scatter(X_max_x[iron_ind:], b_x[iron_ind:], color=COLORS[ind], marker='x')
    ax4.scatter(X_max_y[:iron_ind], b_y[:iron_ind], color=COLORS[ind])
    ax4.scatter(X_max_y[iron_ind:], b_y[iron_ind:], color=COLORS[ind], marker='x')

    ax3.set_xlabel(r"$X_{max}[g/cm^2]$")
    ax3.set_ylabel(r"$b$ (x-component) [1/MHz]")
    ax3.set_title(f"X = {XSLICE} g/cm^2 r = {DISTANCES[ANTENNA] / 100} m")
    ax3.ticklabel_format(axis='y', useMathText=True, scilimits=(0, 0))

    ax4.set_xlabel(r"$X_{max}[g/cm^2]$")
    ax4.set_ylabel(r"$b$ (y-component) [1/MHz]")
    ax4.set_title(f"X = {XSLICE} g/cm^2 r = {DISTANCES[ANTENNA] / 100} m")
    ax4.ticklabel_format(axis='y', useMathText=True, scilimits=(0, 0))

    ax5.scatter(X_max_x[:iron_ind], c_x[:iron_ind], color=COLORS[ind])
    ax5.scatter(X_max_x[iron_ind:], c_x[iron_ind:], color=COLORS[ind], marker='x')
    ax6.scatter(X_max_y[:iron_ind], c_y[:iron_ind], color=COLORS[ind])
    ax6.scatter(X_max_y[iron_ind:], c_y[iron_ind:], color=COLORS[ind], marker='x')

    ax5.set_xlabel(r"$X_{max}[g/cm^2]$")
    ax5.set_ylabel(r"$c$ (x-component) [$1/MHz^2$]")
    ax5.set_title(f"X = {XSLICE} g/cm^2 r = {DISTANCES[ANTENNA] / 100} m")
    ax5.ticklabel_format(axis='y', useMathText=True, scilimits=(0, 0))

    ax6.set_xlabel(r"$X_{max}[g/cm^2]$")
    ax6.set_ylabel(r"$c$ (y-component) [$1/MHz^2$]")
    ax6.set_title(f"X = {XSLICE} g/cm^2 r = {DISTANCES[ANTENNA] / 100} m")
    ax6.ticklabel_format(axis='y', useMathText=True, scilimits=(0, 0))

    A_x_tot.extend(A_0_x)
    A_y_tot.extend(A_0_y)
    b_x_tot.extend(b_x)
    b_y_tot.extend(b_y)
    c_x_tot.extend(c_x)
    c_y_tot.extend(c_y)
    X_max_tot.extend(X_max_x)

x_plot = np.arange(500, 900, 1)
axis = [ax1, ax2, ax3, ax4, ax5, ax6]
plots = [A_x_tot, A_y_tot, b_x_tot, b_y_tot, c_x_tot, c_y_tot]

# Plot David's results on top
if 'david' in PLOT_TYPES:
    param_x_david = []
    param_y_david = []
    names_david = []

    prev = os.getcwd()
    os.chdir(DAVID_DIRECTORY)
    for file in os.listdir('./'):
        names_david.append(file.split('ampfit')[0])
        david_data = np.genfromtxt(file).reshape((207, 6, 2, 3))
        param_x_david.append(david_data[int(XSLICE / 5 - 1), ANTENNA, 0, :])
        param_y_david.append(david_data[int(XSLICE / 5 - 1), ANTENNA, 1, :])
    os.chdir(prev)

    X_max_david = [X_max_dict[key] for key in names_david]
    for ind, ax in enumerate(axis):
        ran = int(ind / 2)
        if ind % 2 == 0:
            ax.scatter(X_max_david[:iron_index[0]], [p[ran] for p in param_x_david[:iron_index[0]]],
                       color=COLORS_DAVID[0])
            ax.scatter(X_max_david[iron_index[0]:len(param_x_david)], [p[ran] for p in param_x_david[iron_index[0]:]],
                       color=COLORS_DAVID[0], marker='x')
        else:
            ax.scatter(X_max_david[:iron_index[0]], [p[ran] for p in param_y_david[:iron_index[0]]],
                       color=COLORS_DAVID[0])
            ax.scatter(X_max_david[iron_index[0]:len(param_y_david)], [p[ran] for p in param_y_david[iron_index[0]:]],
                       color=COLORS_DAVID[0], marker='x')

# if 'polynomial_fit' in FIT_TYPES:
#         poly = np.polynomial.polynomial.polyfit(X_max_tot[:len(param_x_david)], param_x_david, 2)
#         ax1.plot(x_plot, poly[0] + poly[1] * x_plot + poly[2] * x_plot ** 2,
#                  label='Polynomial fit for David', linestyle='--', color=COLORS_DAVID[0])
#
#         poly = np.polynomial.polynomial.polyfit(X_max_tot[:len(param_y_david)], param_y_david, 2)
#         ax2.plot(x_plot, poly[0] + poly[1] * x_plot + poly[2] * x_plot ** 2,
#                  label='Polynomial fit for David', linestyle='--', color=COLORS_DAVID[0])

# Indicate position of target shower
if 'target' in PLOT_TYPES:
    target = target_index[0]
    for ind, ax in enumerate(axis):
        ax.scatter(X_max_tot[target], plots[ind][target], marker='^', color='orangered',
                   s=4 * rcParams['lines.markersize'] ** 2)

# Calculate, fit and plot profiles
if 'profile' in FIT_TYPES:
    bin_edges = np.concatenate((np.arange(np.floor(min(X_max_tot)), 900, 10), [900, 915, 975]))

    for ind, plot in enumerate(plots):
        mean, std = make_profile(bin_edges, X_max_tot, plot)
        fit_params, _ = fit_profile(lambda x, p0, p1, p2: p0 + p1 * x + p2 * x ** 2,
                                    bin_edges, mean, std)
        plot_fit_profile(axis[ind], x_plot, fit_params, bin_edges, mean, std)

# Plot the parabola on top of the figures
if 'file' in FIT_TYPES:
    ax1.plot(x_plot, arX[ANTENNA, 1] + arX[ANTENNA, 2] * x_plot + arX[ANTENNA, 3] * x_plot ** 2, label='Fit from file')
    ax2.plot(x_plot, arY[ANTENNA, 1] + arY[ANTENNA, 2] * x_plot + arY[ANTENNA, 3] * x_plot ** 2, label='Fit from file')

    ax3.plot(x_plot, arX[ANTENNA, 4] + arX[ANTENNA, 5] * x_plot + arX[ANTENNA, 6] * x_plot ** 2, label='Fit from file')
    ax4.plot(x_plot, arY[ANTENNA, 4] + arY[ANTENNA, 5] * x_plot + arY[ANTENNA, 6] * x_plot ** 2, label='Fit from file')

    ax5.plot(x_plot, arX[ANTENNA, 7] + arX[ANTENNA, 8] * x_plot + arX[ANTENNA, 9] * x_plot ** 2, label='Fit from file')
    ax6.plot(x_plot, arY[ANTENNA, 7] + arY[ANTENNA, 8] * x_plot + arY[ANTENNA, 9] * x_plot ** 2, label='Fit from file')

# Fit the tot lists directly and plot them
if 'curve_fit' in FIT_TYPES:
    resX_A, _ = curve_fit(lambda x, p0, p1, p2: p0 + p1 * x + p2 * x ** 2, X_max_tot, A_x_tot)
    resX_b, _ = curve_fit(lambda x, p0, p1, p2: p0 + p1 * x + p2 * x ** 2, X_max_tot, b_x_tot)
    resX_c, _ = curve_fit(lambda x, p0, p1, p2: p0 + p1 * x + p2 * x ** 2, X_max_tot, c_x_tot,
                          p0=[1e-3, 1e-5, 1e-10], sigma=np.array(c_x_tot)*0.1)
    resY_A, _ = curve_fit(lambda x, p0, p1, p2: p0 + p1 * x + p2 * x ** 2, X_max_tot, A_y_tot)
    resY_b, _ = curve_fit(lambda x, p0, p1, p2: p0 + p1 * x + p2 * x ** 2, X_max_tot, b_y_tot)
    resY_c, _ = curve_fit(lambda x, p0, p1, p2: p0 + p1 * x + p2 * x ** 2, X_max_tot, c_y_tot,
                          p0=[1e-3, 1e-5, 1e-10], sigma=np.array(c_y_tot)*0.1)

    ax1.plot(x_plot, resX_A[0] + resX_A[1] * x_plot + resX_A[2] * x_plot ** 2, label='Curve fit')
    ax2.plot(x_plot, resY_A[0] + resY_A[1] * x_plot + resY_A[2] * x_plot ** 2, label='Curve fit')

    ax3.plot(x_plot, resX_b[0] + resX_b[1] * x_plot + resX_b[2] * x_plot ** 2, label='Curve fit')
    ax4.plot(x_plot, resY_b[0] + resY_b[1] * x_plot + resY_b[2] * x_plot ** 2, label='Curve fit')

    ax5.plot(x_plot, resX_c[0] + resX_c[1] * x_plot + resX_c[2] * x_plot ** 2, label='Curve fit')
    ax6.plot(x_plot, resY_c[0] + resY_c[1] * x_plot + resY_c[2] * x_plot ** 2, label='Curve fit')

# Fit scatter using NumPy's polyfit
if 'polynomial_fit' in FIT_TYPES:
    for ax, plot in zip(axis, plots):
        poly = np.polynomial.polynomial.polyfit(X_max_tot, plot, 2)
        ax.plot(x_plot, poly[0] + poly[1] * x_plot + poly[2] * x_plot ** 2, label='Polynomial fit', linestyle='--')

for ind, ax in enumerate(axis):
    ax.legend()
    ax.set_ylim(np.percentile(plots[ind], 5)-3*np.std(plots[ind]), np.percentile(plots[ind], 95)+3*np.std(plots[ind]))

# Make the figures active and save them
os.chdir('../Figures')

plt.figure(fig1)
plt.savefig(f'A0_{ANTENNA}x{XSLICE}.png', bbox_inches='tight')

plt.figure(fig2)
plt.savefig(f'b_{ANTENNA}x{XSLICE}.png', bbox_inches='tight')

plt.figure(fig3)
plt.savefig(f'c_{ANTENNA}x{XSLICE}.png', bbox_inches='tight')
