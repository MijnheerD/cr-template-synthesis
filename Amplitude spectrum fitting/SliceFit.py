import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.optimize import curve_fit
from scipy.constants import c as c_vacuum


def get_number_of_particles(path, slice):
    long = np.genfromtxt(path, skip_header=2, skip_footer=216, usecols=(2,3))
    return np.sum(long[int(slice/5 + 1), :])


def amplitude_fun_slice(f, a_0, b, c, d, f0=0):
    return a_0 * np.exp(b * (f - f0) + c * (f - f0) ** 2) + d


def amplitude_fit_slice(fdata, ydata, x, r, f0=0):
    d = max(0, 1e-9 * (x/400 - 1.5) * np.exp(1 - r/40000))
    try:
        popt, pcov = curve_fit(lambda f, a_0, b, c: a_0 * np.exp(b * (f - f0) + c * (f - f0) ** 2), fdata / 1e6,
                               ydata - d**2, p0=[ydata[0], -2.5e-3, 0], bounds=(-np.inf, [np.inf, 1e-2, 1e-3]),
                               sigma=np.maximum(ydata*0.1, np.maximum(d**2, 1e-18)))
    except RuntimeError:
        popt, pcov = curve_fit(lambda f, a_0, b, c: a_0 + b * (f - f0) + c * (f - f0)**2, fdata/1e6,
                               np.log(ydata - d**2), p0=[np.log(25), -1e-3, -1e-6])
        popt[0] = np.exp(popt[0])  # The first variable in the linear fit is ln(A_0)
    return popt, d**2


distances = [1, 4000, 7500, 11000, 15000, 37500]
files_path = '/mnt/hgfs/Shared data/BulkSynth/bulksynth-18/'
david_path = '/mnt/hgfs/Shared data/ampfitQ/'
sim = '110002'
antenna = 3
Xslice = 655

with open(files_path+'SIM'+sim+f'_coreas/raw_{antenna}x{Xslice}.dat', 'r') as file:
    data = np.genfromtxt(file) * c_vacuum * 1e2  # conversion from statV/cm to microV/m

freq = np.fft.rfftfreq(len(data), 2e-10)
frange = np.logical_and(12 * 1e6 <= freq, freq <= 502 * 1e6)

Nslice = get_number_of_particles(files_path+'DAT' + sim + '.long', Xslice)
spectrum = np.apply_along_axis(np.fft.rfft, 0, data[:, 1:], norm='forward')
amplitude = np.abs(spectrum) * 2  # Need to multiply by 2 because 1-sided FT
filtered = amplitude[frange]
'''
plt.plot(freq[frange]/1e6, filtered[:, 0], label='x-component (CE)')
plt.plot(freq[frange]/1e6, filtered[:, 1], label='y-component (Geo)')
#plt.ylim([0, 8.5])
plt.xlabel('f [MHz]')
plt.ylabel(r'A [\mu V / m]$')
plt.title(f'r = {distances[antenna]/100}m, Xslice = {Xslice} g/cm^2')
plt.legend()
plt.show()
'''
coefX, dX = amplitude_fit_slice(freq[frange],  filtered[:, 0], Xslice, distances[antenna], f0=250)
coefY, dY = amplitude_fit_slice(freq[frange],  filtered[:, 1], Xslice, distances[antenna], f0=250)

# coef0X, d0X = amplitude_fit_slice(freq[frange]-2.5*1e8, ampXfiltered, Xslice, distances[antenna])
# coef0Y, d0Y = amplitude_fit_slice(freq[frange]-2.5*1e8, ampYfiltered, Xslice, distances[antenna])

print('The x component has parameters', *coefX)
print('The y component has parameters', *coefY)
''' 
david_data = np.genfromtxt(david_path + sim + 'ampfit_Q.txt')
david_data = david_data.reshape((207, 6, 2, 3))

print('David x component has parameters', *david_data[int(Xslice/5 - 1), antenna, 0, :])
print('David y component has parameters', *david_data[int(Xslice/5 - 1), antenna, 1, :])
'''
fig = plt.figure()

ax = fig.add_subplot(121)
ax.plot(freq[frange]/1e6,  filtered[:, 0], linestyle='--', color='orange')
ax.plot(freq[frange]/1e6, amplitude_fun_slice(freq[frange]/1e6, (coefX[0]), coefX[1], coefX[2], dX, f0=250),
        color='orange', label='My fit')
# ax.plot(freq[frange]/1e6, amplitude_fun_slice(freq[frange]/1e6, *david_data[int(Xslice/5 - 1), antenna, 0, :], dX-0.11e-17),
#         linestyle='dotted', color='black', label='David result')
ax.set_xlabel('Frequency (MHz)')
#ax.set_yscale('log')
ax.set_title('The x-component')
ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
# ax.ticklabel_format(scilimits=(-10, -10), axis='y')
ax.legend()

ax2 = fig.add_subplot(122)
ax2.plot(freq[frange]/1e6,  filtered[:, 1], linestyle='--', color='blue')
ax2.plot(freq[frange]/1e6, amplitude_fun_slice(freq[frange]/1e6, (coefY[0]), coefY[1], coefY[2], dY, f0=250),
         color='blue', label='My fit')
# ax2.plot(freq[frange]/1e6, amplitude_fun_slice(freq[frange]/1e6, *david_data[int(Xslice/5 - 1), antenna, 1, :], dY-0.23e-17),
#          linestyle='dotted', color='black', label='David result')
ax2.set_xlabel('Frequency (MHz)')
ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
# ax2.ticklabel_format(scilimits=(-10, -10), axis='y')
ax2.set_title('The y-component')
ax2.legend()

plt.show()
