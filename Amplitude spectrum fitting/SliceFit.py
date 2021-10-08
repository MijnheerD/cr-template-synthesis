import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.optimize import curve_fit
from scipy.constants import c as c_vacuum


def amplitude_fun_slice(f, a_0, b, c, d):
    return a_0 * np.exp(b * f + c * f ** 2) + d


def amplitude_fit_slice(fdata, ydata, x, r):
    d = max(0, 1e-9 * (x/400 - 1.5) * np.exp(1 - r/40000))
    try:
        popt, pcov = curve_fit(lambda f, a_0, b, c: a_0 * np.exp(b * f + c * f ** 2), fdata/1e6, ydata - d**2,
                               p0=[4e-9, -1e-3, -1e-4])
    except RuntimeError:
        popt, pcov = curve_fit(lambda f, a_0, b, c: a_0 + b * f + c * f**2, fdata/1e6, np.log(ydata - d**2),
                               p0=[np.log(4e-9), -1e-3, -1e-6])
        popt[0] = np.exp(popt[0])  # The first variable in the linear fit is ln(A_0)
    return popt, d**2


distances = [1, 4000, 7500, 11000, 15000, 37500]
files_path = '/mnt/hgfs/Shared data/BulkSynth/bulksynth-7/'
sim = 'SIM100002'  # shower with Xmax = 862 g/cm2
antenna = 3
Xslice = 655

with open(files_path+sim+f'_coreas/raw_{antenna}x{Xslice}.dat', 'r') as file:
    data = np.genfromtxt(file) * c_vacuum * 1e2  # conversion from statV/cm to microV/m

freq = np.fft.rfftfreq(len(data), 2e-10)
frange = np.logical_and(10 * 1e6 <= freq, freq <= 5 * 1e8)

spectrum = np.apply_along_axis(np.fft.rfft, 0, data[:, 1:])
amplitude = np.abs(spectrum)  # Need to multiply by 2 because 1-sided FT
filtered = amplitude[frange]

plt.plot(freq[frange]/1e6, filtered[:, 0]/35863386*1e7, label='x-component (CE)')
plt.plot(freq[frange]/1e6, filtered[:, 1]/35863386*1e7, label='y-component (Geo)')
plt.ylim([0, 8.5])
plt.xlabel('f [MHz]')
plt.ylabel(r'$10^7 \cdot A / N(X_{slice}) [\mu V / m]$')
plt.title(f'r = {distances[antenna]/100}m, Xslice = {Xslice} g/cm^2')
plt.legend()
plt.show()

coefX, dX = amplitude_fit_slice(freq[filter], ampXfiltered, Xslice, distances[antenna])
coefY, dY = amplitude_fit_slice(freq[filter], ampYfiltered, Xslice, distances[antenna])

# coef0X, d0X = amplitude_fit_slice(freq[filter]-2.5*1e8, ampXfiltered, Xslice, distances[antenna])
# coef0Y, d0Y = amplitude_fit_slice(freq[filter]-2.5*1e8, ampYfiltered, Xslice, distances[antenna])

print('The x component has parameters', *coefX)
print('The y component has parameters', *coefY)

fig = plt.figure()

ax = fig.add_subplot(121)
ax.plot(freq[filter]/1e6, ampXfiltered, linestyle='--', color='orange')
ax.plot(freq[filter]/1e6, amplitude_fun_slice(freq[filter]/1e6, (coefX[0]), coefX[1], coefX[2], dX),
        color='orange')
# ax.plot((freq[filter])/1e6, amplitude_fun_slice(freq[filter]-2.5*1e8, np.exp(coef0X[0]), coef0X[1], coef0X[2], d0X),
#         linestyle='dotted', color='orange')
ax.set_xlabel('Frequency (MHz)')
#ax.set_yscale('log')
ax.set_title('The x-component')
ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.ticklabel_format(scilimits=(-10, -10), axis='y')

ax2 = fig.add_subplot(122)
ax2.plot(freq[filter]/1e6, ampYfiltered, linestyle='--', color='blue')
ax2.plot(freq[filter]/1e6, amplitude_fun_slice(freq[filter]/1e6, (coefY[0]), coefY[1], coefY[2], dY),
         color='blue')
ax2.set_xlabel('Frequency (MHz)')
ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax2.ticklabel_format(scilimits=(-10, -10), axis='y')
ax2.set_title('The y-component')

plt.show()
