import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


FIT_DIRECTORY = 'fitFiles17/'  # location of the files containing the fit parameters
DISTANCES = [1, 4000, 7500, 11000, 15000, 37500]  # antenna radial distances to shower core, in cm
XSLICE = 855
ANTENNA = 3

A_0_x = []
b_x = []
c_x = []
X_max_x = []

A_0_y = []
b_y = []
c_y = []
X_max_y = []

for file in os.listdir(os.path.join(FIT_DIRECTORY, 'fitX')):
    with open(os.path.join(FIT_DIRECTORY, 'fitX', file), 'r') as fX, open(os.path.join(FIT_DIRECTORY, 'fitY', file), 'r') as fY:
        for lineX, lineY in zip(fX, fY):
            lstX = lineX.split(', ')
            lstY = lineY.split(', ')
            if float(lstX[0]) == XSLICE:
                if float(lstX[1]) == ANTENNA:
                    A_0_x.append(float(lstX[2]))
                    b_x.append(float(lstX[3]))
                    c_x.append(float(lstX[4]))
                    X_max_x.append(float(lstX[5]))
                    A_0_y.append(float(lstY[2]))
                    b_y.append(float(lstY[3]))
                    c_y.append(float(lstY[4]))
                    X_max_y.append(float(lstY[5]))
                    break  # We know there is only 1 interesting entry per file
                    
resX, covX = curve_fit(lambda x, p0, p1, p2: p0 + p1 * x + p2 * x**2, X_max_x, A_0_x)
resY, covY = curve_fit(lambda x, p0, p1, p2: p0 + p1 * x + p2 * x**2, X_max_y, A_0_y)
x_plot = np.array(X_max_y)
x_plot.sort()

# plt.style.use('dark_background')
fig1, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 6))

ax1.scatter(X_max_x[:100], A_0_x[:100], color='cyan')
ax1.scatter(X_max_x[100:], A_0_x[100:], color='cyan', marker='x')
ax1.axvline(x=XSLICE, linestyle='--', label=r'$X_{slice}$')
ax1.axvline(x=-resX[1]/(2*resX[2]), linestyle=':', label='Top of parabola')
ax1.plot(x_plot, resX[0] + resX[1] * x_plot + resX[2] * x_plot**2)
ax1.legend()

ax2.scatter(X_max_y[:100], A_0_y[:100], color='cyan')
ax2.scatter(X_max_y[100:], A_0_y[100:], color='cyan', marker='x')
ax2.axvline(x=XSLICE, linestyle='--', label=r'$X_{slice}$')
ax2.axvline(x=-resY[1]/(2*resY[2]), linestyle=':', label='Top of parabola')
ax2.plot(x_plot, resY[0] + resY[1] * x_plot + resY[2] * x_plot**2)
ax2.legend()

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

plt.show()
'''
plt.savefig(f'A0_{ANTENNA}x{XSLICE}_1e17.png', bbox_inches='tight')

fig2, [ax3, ax4] = plt.subplots(1, 2, figsize=(12, 6))

ax3.scatter(X_max_x[:100], b_x[:100], color='cyan')
ax3.scatter(X_max_x[100:], b_x[100:], color='cyan', marker='x')
ax4.scatter(X_max_y[:100], b_y[:100], color='cyan')
ax4.scatter(X_max_y[100:], b_y[100:], color='cyan', marker='x')

ax3.set_xlabel(r"$X_{max}[g/cm^2]$")
ax3.set_ylabel(r"$b$ (x-component) [1/MHz]")
ax3.set_title(f"X = {XSLICE} g/cm^2 r = {DISTANCES[ANTENNA] / 100} m")
ax3.ticklabel_format(axis='y', useMathText=True, scilimits=(0, 0))

ax4.set_xlabel(r"$X_{max}[g/cm^2]$")
ax4.set_ylabel(r"$b$ (y-component) [1/MHz]")
ax4.set_title(f"X = {XSLICE} g/cm^2 r = {DISTANCES[ANTENNA] / 100} m")
ax4.ticklabel_format(axis='y', useMathText=True, scilimits=(0, 0))

plt.savefig(f'b_{ANTENNA}x{XSLICE}_1e17.png', bbox_inches='tight')

fig3, [ax5, ax6] = plt.subplots(1, 2, figsize=(12, 6))

ax5.scatter(X_max_x[:100], c_x[:100], color='cyan')
ax5.scatter(X_max_x[100:], c_x[100:], color='cyan', marker='x')
ax6.scatter(X_max_y[:100], c_y[:100], color='cyan')
ax6.scatter(X_max_y[100:], c_y[100:], color='cyan', marker='x')

ax5.set_xlabel(r"$X_{max}[g/cm^2]$")
ax5.set_ylabel(r"$c$ (x-component) [$1/MHz^2$]")
ax5.set_title(f"X = {XSLICE} g/cm^2 r = {DISTANCES[ANTENNA] / 100} m")
ax5.ticklabel_format(axis='y', useMathText=True, scilimits=(0, 0))

ax6.set_xlabel(r"$X_{max}[g/cm^2]$")
ax6.set_ylabel(r"$c$ (y-component) [$1/MHz^2$]")
ax6.set_title(f"X = {XSLICE} g/cm^2 r = {DISTANCES[ANTENNA] / 100} m")
ax6.ticklabel_format(axis='y', useMathText=True, scilimits=(0, 0))

plt.savefig(f'c_{ANTENNA}x{XSLICE}_1e17.png', bbox_inches='tight')
'''