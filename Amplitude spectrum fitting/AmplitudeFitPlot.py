import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit


def get_number_of_particles(path, x_slice):
    long = np.genfromtxt(path, skip_header=2, skip_footer=216, usecols=(2, 3))
    return np.sum(long[int(x_slice / 5 - 1), :])


def make_profile(bins, x_data, y_data):
    mean = binned_statistic(x_data, y_data, statistic='mean', bins=len(bins) - 1, range=(min(bins), max(bins)))
    std = binned_statistic(x_data, y_data, statistic='std', bins=len(bins) - 1, range=(min(bins), max(bins)))
    return mean[0], std[0]


def fit_profile(fitfun, bins, mean, std):
    bin_centers = (bins[1:] + bins[:-1]) / 2
    to_fit = list(zip(*[(bin_centers[i], mean[i], std[i]) for i in range(len(bin_centers)) if std[i] != 0]))
    popt, pcov = curve_fit(fitfun,  to_fit[0], to_fit[1],
                           p0=[np.median(to_fit[1]), 1e-8, 1e-11], sigma=to_fit[2])
    return popt, pcov


def plot_fit_profile(ax, x_plot, param, bins, mean, std, bar=False):
    width = int(bins[1] - bins[0])
    ax.plot(x_plot, param[0] + param[1] * x_plot + param[2] * x_plot ** 2, label='Fit to profile')
    if bar:
        ax.bar(bins[:-1], mean, align='edge', width=width, color='w', alpha=0.3, yerr=std, ecolor='w')


FIT_DIRECTORIES = ['fitFiles5017', 'fitFiles5018', 'fitFiles5019']  # location of the files containing fit parameters
COLORS = ['cyan', 'magenta', 'yellow']
REAS_DIRECTORY = '/mnt/hgfs/Shared data/BulkSynth/CORSIKA_long_files'
PARAM_DIRECTORY = 'paramProfileFit50'
DISTANCES = [1, 4000, 7500, 11000, 15000, 37500]  # antenna_nr radial distances to shower core, in cm
XSLICE = 25
ANTENNA = 0

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

for ind, directory in enumerate(FIT_DIRECTORIES):
    A_0_x = []
    b_x = []
    c_x = []
    X_max_x = []

    A_0_y = []
    b_y = []
    c_y = []
    X_max_y = []
    for file in os.listdir(os.path.join(directory, 'fitX')):
        n_slice = get_number_of_particles(os.path.join(REAS_DIRECTORY,
                                                       'DAT' + file.split('.')[0].split('SIM')[1] + '.long'), XSLICE)
        if n_slice == 0:
            n_slice = 1
        with open(os.path.join(directory, 'fitX', file), 'r') as fX, \
                open(os.path.join(directory, 'fitY', file), 'r') as fY:
            for lineX, lineY in zip(fX, fY):
                lstX = lineX.split(', ')
                lstY = lineY.split(', ')
                if float(lstX[0]) == XSLICE:
                    if float(lstX[1]) == ANTENNA:
                        A_0_x.append(float(lstX[2]) / n_slice)
                        b_x.append(float(lstX[3]))
                        c_x.append(float(lstX[4]))
                        X_max_x.append(float(lstX[5]))
                        A_0_y.append(float(lstY[2]) / n_slice)
                        b_y.append(float(lstY[3]))
                        c_y.append(float(lstY[4]))
                        X_max_y.append(float(lstY[5]))
                        break  # We know there is only 1 interesting entry per file

    ax1.scatter(X_max_x[:100], A_0_x[:100], color=COLORS[ind])
    ax1.scatter(X_max_x[100:], A_0_x[100:], color=COLORS[ind], marker='x')
    ax2.scatter(X_max_y[:100], A_0_y[:100], color=COLORS[ind])
    ax2.scatter(X_max_y[100:], A_0_y[100:], color=COLORS[ind], marker='x')

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

    ax3.scatter(X_max_x[:100], b_x[:100], color=COLORS[ind])
    ax3.scatter(X_max_x[100:], b_x[100:], color=COLORS[ind], marker='x')
    ax4.scatter(X_max_y[:100], b_y[:100], color=COLORS[ind])
    ax4.scatter(X_max_y[100:], b_y[100:], color=COLORS[ind], marker='x')

    ax3.set_xlabel(r"$X_{max}[g/cm^2]$")
    ax3.set_ylabel(r"$b$ (x-component) [1/MHz]")
    ax3.set_title(f"X = {XSLICE} g/cm^2 r = {DISTANCES[ANTENNA] / 100} m")
    ax3.ticklabel_format(axis='y', useMathText=True, scilimits=(0, 0))

    ax4.set_xlabel(r"$X_{max}[g/cm^2]$")
    ax4.set_ylabel(r"$b$ (y-component) [1/MHz]")
    ax4.set_title(f"X = {XSLICE} g/cm^2 r = {DISTANCES[ANTENNA] / 100} m")
    ax4.ticklabel_format(axis='y', useMathText=True, scilimits=(0, 0))

    ax5.scatter(X_max_x[:100], c_x[:100], color=COLORS[ind])
    ax5.scatter(X_max_x[100:], c_x[100:], color=COLORS[ind], marker='x')
    ax6.scatter(X_max_y[:100], c_y[:100], color=COLORS[ind])
    ax6.scatter(X_max_y[100:], c_y[100:], color=COLORS[ind], marker='x')

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

# Calculate, fit and plot profiles
bin_edges = np.arange(min(X_max_tot), max(X_max_tot) + 10, 10)

for ind, plot in enumerate(plots):
    mean, std = make_profile(bin_edges, X_max_tot, plot)
    fit_params, _ = fit_profile(lambda x, p0, p1, p2: p0 + p1 * x + p2 * x ** 2,
                                bin_edges, mean, std)
    plot_fit_profile(axis[ind], x_plot, fit_params, bin_edges, mean, std)

# Plot the parabola on top of the figures
ax1.plot(x_plot, arX[ANTENNA, 1] + arX[ANTENNA, 2] * x_plot + arX[ANTENNA, 3] * x_plot ** 2, label='Fit from file')
ax2.plot(x_plot, arY[ANTENNA, 1] + arY[ANTENNA, 2] * x_plot + arY[ANTENNA, 3] * x_plot ** 2, label='Fit from file')

ax3.plot(x_plot, arX[ANTENNA, 4] + arX[ANTENNA, 5] * x_plot + arX[ANTENNA, 6] * x_plot ** 2, label='Fit from file')
ax4.plot(x_plot, arY[ANTENNA, 4] + arY[ANTENNA, 5] * x_plot + arY[ANTENNA, 6] * x_plot ** 2, label='Fit from file')

ax5.plot(x_plot, arX[ANTENNA, 7] + arX[ANTENNA, 8] * x_plot + arX[ANTENNA, 9] * x_plot ** 2, label='Fit from file')
ax6.plot(x_plot, arY[ANTENNA, 7] + arY[ANTENNA, 8] * x_plot + arY[ANTENNA, 9] * x_plot ** 2, label='Fit from file')

# Fit the tot lists directly and plot them
resX_A, _ = curve_fit(lambda x, p0, p1, p2: p0 + p1 * x + p2 * x ** 2, X_max_tot, A_x_tot)
resX_b, _ = curve_fit(lambda x, p0, p1, p2: p0 + p1 * x + p2 * x ** 2, X_max_tot, b_x_tot)
resX_c, _ = curve_fit(lambda x, p0, p1, p2: p0 + p1 * x + p2 * x ** 2, X_max_tot, c_x_tot,
                      p0=[1e-3, 1e-5, 1e-10], sigma=np.array(c_x_tot)*0.1)
resY_A, _ = curve_fit(lambda x, p0, p1, p2: p0 + p1 * x + p2 * x ** 2, X_max_tot, A_y_tot)
resY_b, _ = curve_fit(lambda x, p0, p1, p2: p0 + p1 * x + p2 * x ** 2, X_max_tot, b_y_tot)
resY_c, _ = curve_fit(lambda x, p0, p1, p2: p0 + p1 * x + p2 * x ** 2, X_max_tot, c_y_tot,
                      p0=[1e-3, 1e-5, 1e-10], sigma=np.array(c_y_tot)*0.1)

ax1.plot(x_plot, resX_A[0] + resX_A[1] * x_plot + resX_A[2] * x_plot ** 2, label='Fit to scatter')
ax2.plot(x_plot, resY_A[0] + resY_A[1] * x_plot + resY_A[2] * x_plot ** 2, label='Fit to scatter')

ax3.plot(x_plot, resX_b[0] + resX_b[1] * x_plot + resX_b[2] * x_plot ** 2, label='Fit to scatter')
ax4.plot(x_plot, resY_b[0] + resY_b[1] * x_plot + resY_b[2] * x_plot ** 2, label='Fit to scatter')

ax5.plot(x_plot, resX_c[0] + resX_c[1] * x_plot + resX_c[2] * x_plot ** 2, label='Fit to scatter')
ax6.plot(x_plot, resY_c[0] + resY_c[1] * x_plot + resY_c[2] * x_plot ** 2, label='Fit to scatter')

for ind, ax in enumerate(axis):
    ax.legend()
    ax.set_ylim(np.percentile(plots[ind], 5)-3*np.std(plots[ind]), np.percentile(plots[ind], 95)+3*np.std(plots[ind]))

# Make the figures active and save them
plt.figure(fig1)
plt.savefig(f'A0_{ANTENNA}x{XSLICE}.png', bbox_inches='tight')

plt.figure(fig2)
plt.savefig(f'b_{ANTENNA}x{XSLICE}.png', bbox_inches='tight')

plt.figure(fig3)
plt.savefig(f'c_{ANTENNA}x{XSLICE}.png', bbox_inches='tight')
