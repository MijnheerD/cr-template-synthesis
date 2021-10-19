import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.constants import c


def get_number_of_particles(path, x_slice):
    long = np.genfromtxt(path, skip_header=2, skip_footer=216, usecols=(2, 3))
    return np.sum(long[int(x_slice / 5 + 1), :])


FIT_DIRECTORIES = ['fitFiles17/', 'fitFiles18/', 'fitFiles19/']  # location of the files containing the fit parameters
COLORS = ['cyan', 'magenta', 'yellow']
REAS_DIRECTORY = ['/mnt/hgfs/Shared data/BulkSynth/bulksynth-17/',
                  '/mnt/hgfs/Shared data/BulkSynth/bulksynth-18/',
                  '/mnt/hgfs/Shared data/BulkSynth/bulksynth-19/']
PARAM_DIRECTORY = '/home/mdesmet/PycharmProjects/cr-template-synthesis/Amplitude spectrum fitting/paramFit/'
DISTANCES = [1, 4000, 7500, 11000, 15000, 37500]  # antenna radial distances to shower core, in cm
XSLICE = 655
ANTENNA = 3

arX = np.genfromtxt(os.path.join(PARAM_DIRECTORY, 'fitX', 'slice' + str(XSLICE) + '.dat'))
arY = np.genfromtxt(os.path.join(PARAM_DIRECTORY, 'fitY', 'slice' + str(XSLICE) + '.dat'))

# Make the figures to plot the parameters
# plt.style.use('dark_background')
fig1, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 6))
fig2, [ax3, ax4] = plt.subplots(1, 2, figsize=(12, 6))
fig3, [ax5, ax6] = plt.subplots(1, 2, figsize=(12, 6))

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
        n_slice = get_number_of_particles(os.path.join(REAS_DIRECTORY[ind],
                                                       'DAT' + file.split('.')[0].split('SIM')[1] + '.long'), XSLICE)
        with open(os.path.join(directory, 'fitX', file), 'r') as fX, \
                open(os.path.join(directory, 'fitY', file), 'r') as fY:
            for lineX, lineY in zip(fX, fY):
                lstX = lineX.split(', ')
                lstY = lineY.split(', ')
                if float(lstX[0]) == XSLICE:
                    if float(lstX[1]) == ANTENNA:
                        A_0_x.append(float(lstX[2]) / n_slice / c / 1e2)
                        b_x.append(float(lstX[3]))
                        c_x.append(float(lstX[4]))
                        X_max_x.append(float(lstX[5]))
                        A_0_y.append(float(lstY[2]) / n_slice / c / 1e2)
                        b_y.append(float(lstY[3]))
                        c_y.append(float(lstY[4]))
                        X_max_y.append(float(lstY[5]))
                        break  # We know there is only 1 interesting entry per file

    ax1.scatter(X_max_x[:100], A_0_x[:100], color=COLORS[ind])
    ax1.scatter(X_max_x[100:], A_0_x[100:], color=COLORS[ind], marker='x')
    ax2.scatter(X_max_y[:100], A_0_y[:100], color=COLORS[ind])
    ax2.scatter(X_max_y[100:], A_0_y[100:], color=COLORS[ind], marker='x')

    ax1.set_xlabel(r"$X_{max}[g/cm^2]$")
    ax1.set_ylabel(r"$A_0$ (x-component) [a.u.]")
    ax1.set_xlim([500, 950])
    ax1.set_title(f"X = {XSLICE} g/cm^2 r = {DISTANCES[ANTENNA] / 100} m")
    ax1.ticklabel_format(axis='y', useMathText=True, scilimits=(0, 0))

    ax2.set_xlabel(r"$X_{max}[g/cm^2]$")
    ax2.set_ylabel(r"$A_0$ (y-component) [a.u.]")
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

x_plot = np.arange(500, 900, 1)

# ax1.axvline(x=XSLICE, linestyle='--', label=r'$X_{slice}$')
# ax1.axvline(x=-resX[1]/(2*resX[2]), linestyle=':', label='Top of parabola')
# ax1.legend()

# Plot the parabola on top of the figures
ax1.plot(x_plot, (arX[ANTENNA, 1] + arX[ANTENNA, 2] * x_plot + arX[ANTENNA, 3] * x_plot**2) / c / 1e2)
ax2.plot(x_plot, (arY[ANTENNA, 1] + arY[ANTENNA, 2] * x_plot + arY[ANTENNA, 3] * x_plot**2) / c / 1e2)

ax3.plot(x_plot, arX[ANTENNA, 4] + arX[ANTENNA, 5] * x_plot + arX[ANTENNA, 6] * x_plot**2)
ax4.plot(x_plot, arY[ANTENNA, 4] + arY[ANTENNA, 5] * x_plot + arY[ANTENNA, 6] * x_plot**2)

ax5.plot(x_plot, arX[ANTENNA, 7] + arX[ANTENNA, 8] * x_plot + arX[ANTENNA, 9] * x_plot**2)
ax6.plot(x_plot, arY[ANTENNA, 7] + arY[ANTENNA, 8] * x_plot + arY[ANTENNA, 9] * x_plot**2)

# Make the figures active and save them
plt.figure(fig1)
plt.savefig(f'A0_{ANTENNA}x{XSLICE}.png', bbox_inches='tight')

plt.figure(fig2)
plt.savefig(f'b_{ANTENNA}x{XSLICE}.png', bbox_inches='tight')

plt.figure(fig3)
plt.savefig(f'c_{ANTENNA}x{XSLICE}.png', bbox_inches='tight')
