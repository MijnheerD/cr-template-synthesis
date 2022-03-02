import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.optimize import curve_fit
from scipy.constants import c as c_vacuum


def get_number_of_particles(path, slice):
    long = np.genfromtxt(path, skip_header=2, skip_footer=216, usecols=(2, 3))
    return np.sum(long[int(slice/5 + 1), :])


def amplitude_fun_slice(f, a_0, b, c, d, f0=0):
    return a_0 * np.exp(b * (f - f0) + c * (f - f0) ** 2) + d


def amplitude_fit_slice(fdata, ydata, x, r, f0=0):
    d = max(0, 1e-9 * (x/400 - 1.5) * np.exp(1 - r/40000))
    try:
        popt, pcov = curve_fit(lambda f, a_0, b, c: a_0 * np.exp(b * (f - f0) + c * (f - f0) ** 2), fdata / 1e6,
                               ydata - d**2, p0=[ydata[0], -2.5e-3, 0], bounds=(-np.inf, [np.inf, 1e-2, 1e-3]),
                               sigma=np.ones(ydata.shape)*np.maximum(d**2, 1e-18))
    except RuntimeError:
        popt, pcov = curve_fit(lambda f, a_0, b, c: a_0 + b * (f - f0) + c * (f - f0)**2, fdata/1e6,
                               np.log(ydata - d**2), p0=[np.log(25), -1e-3, -1e-6])
        popt[0] = np.exp(popt[0])  # The first variable in the linear fit is ln(A_0)
    return popt, d**2


def amplitude_fit_slice_iter(fdata, ydata, x, r, f0=0):
    d = max(0, 1e-9 * (x / 400 - 1.5) * np.exp(1 - r / 40000))
    popt, pcov = curve_fit(lambda f, a_0, b: a_0 * np.exp(b * (f - f0)), fdata / 1e6,
                           ydata - d ** 2, p0=[ydata[0], -1e-3], bounds=(-np.inf, [np.inf, 1e-2]),
                           sigma=np.maximum(ydata * 0.1, np.maximum(d ** 2, 1e-18)))

    qopt, qcov = curve_fit(lambda f, b, c: popt[0] * np.exp(b * (f - f0) + c * (f - f0) ** 2),
                           fdata / 1e6, ydata - d ** 2, bounds=(-np.inf, [1e2, 1e3]),
                           p0=[popt[1], -2e-6], sigma=np.maximum(ydata * 0.1, np.maximum(d ** 2, 1e-18)))

    # qopt[-2] = qopt[-2] / 1e4
    # qopt[-1] = qopt[-1] / 1e6

    return (popt[0], *qopt), d**2


def amplitude_fit_slice_unbound(fdata, ydata, x, r, f0=0):
    d = max(0, 1e-9 * (x / 400 - 1.5) * np.exp(1 - r / 40000))
    try:
        popt, pcov = curve_fit(lambda f, a_0, b, c: a_0 * np.exp(b * (f - f0) + c * (f - f0) ** 2), fdata / 1e6,
                               ydata - d ** 2, p0=[ydata[0], -1e-3, -1e-4], maxfev=10000,
                               sigma=np.maximum(ydata*0.1, np.maximum(d**2, 1e-18)))
    except RuntimeError:
        popt, pcov = curve_fit(lambda f, a_0, b, c: a_0 + b * (f - f0) + c * (f - f0) ** 2, fdata / 1e6,
                               np.log(ydata - d ** 2), p0=[np.log(ydata[0]), -1e-3, -1e-6],
                               sigma=np.log(np.maximum(ydata*0.1, np.maximum(d**2, 1e-18))))
        popt[0] = np.exp(popt[0])  # The first variable in the linear fit is ln(A_0)
    return popt, d ** 2


distances = [1, 4000, 7500, 11000, 15000, 37500]
files_path = '/mnt/hgfs/Shared data/BulkSynth/bulksynth-17/'
david_path = '/mnt/hgfs/Shared data/ampfitQ/'
sim = '100099'
antenna = 3
Xslice = 600

with open(files_path+'SIM'+sim+f'_coreas/raw_{antenna}x{Xslice}.dat', 'r') as file:
    data = np.genfromtxt(file) * c_vacuum * 1e2  # conversion from statV/cm to microV/m

# data = data[:-1800, :]
freq = np.fft.rfftfreq(len(data), 2e-10)
frange = np.logical_and(30 * 1e6 <= freq, freq <= 80 * 1e6)
F0 = 50

Nslice = get_number_of_particles(files_path+'DAT' + sim + '.long', Xslice)
spectrum = np.apply_along_axis(np.fft.rfft, 0, data[:, 1:], norm='forward')
amplitude = np.abs(spectrum) * 2  # Need to multiply by 2 because 1-sided FT
filtered = amplitude[frange]

coefX, dX = amplitude_fit_slice_unbound(freq[frange],  filtered[:, 0], Xslice, distances[antenna], f0=F0)
coefY, dY = amplitude_fit_slice_unbound(freq[frange],  filtered[:, 1], Xslice, distances[antenna], f0=F0)

qoefX = np.polynomial.polynomial.polyfit(freq[frange] / 1e6 - F0,  np.log(filtered[:, 0] - dX), 2)
qoefY = np.polynomial.polynomial.polyfit(freq[frange] / 1e6 - F0,  np.log(filtered[:, 1] - dY), 2)

print('The x component has parameters', *coefX)
print('The y component has parameters', *coefY)

qoefX[0] = np.exp(qoefX[0])
qoefY[0] = np.exp(qoefY[0])

print('The x component has parameters', *qoefX)
print('The y component has parameters', *qoefY)

fig = plt.figure()

ax = fig.add_subplot(121)
ax.plot(freq[frange]/1e6,  filtered[:, 0], linestyle='--', color='orange')
ax.plot(freq[frange]/1e6, amplitude_fun_slice(freq[frange]/1e6, *coefX, dX, f0=F0),
        color='orange', label='My fit')
ax.plot(freq[frange]/1e6, amplitude_fun_slice(freq[frange]/1e6, *qoefX, dX, f0=F0),
        color='yellow', label='Numpy polyfit')
ax.set_xlabel('Frequency (MHz)')
ax.set_yscale('log')
ax.set_title('The x-component')
# ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
# ax.ticklabel_format(scilimits=(-10, -10), axis='y')
ax.legend()

ax2 = fig.add_subplot(122)
ax2.plot(freq[frange]/1e6,  filtered[:, 1], linestyle='--', color='blue')
ax2.plot(freq[frange]/1e6, amplitude_fun_slice(freq[frange]/1e6, *coefY, dY, f0=F0),
         color='blue', label='My fit')
ax2.plot(freq[frange]/1e6, amplitude_fun_slice(freq[frange]/1e6, *qoefY, dY, f0=F0),
         color='lightblue', label='Numpy polyfit')
ax2.set_xlabel('Frequency (MHz)')
ax2.set_yscale('log')
# ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
# ax2.ticklabel_format(scilimits=(-10, -10), axis='y')
ax2.set_title('The y-component')
ax2.legend()

plt.show()
