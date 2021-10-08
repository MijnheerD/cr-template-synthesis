import os
import glob
import time
import numpy as np
from scipy.optimize import curve_fit


def amplitude_fit_slice(fdata, ydata, x, r):
    d = max(0, 1e-9 * (x/400 - 1.5) * np.exp(1 - r/40000))
    try:
        popt, pcov = curve_fit(lambda f, a_0, b, c: a_0 * np.exp(b * f + c * f ** 2), fdata/1e6, ydata - d**2,
                               p0=[25, -1e-3, -1e-4], bounds=(-np.inf, [np.inf, 1e-2, 1e-3]))
    except RuntimeError:
        popt, pcov = curve_fit(lambda f, a_0, b, c: a_0 + b * f + c * f**2, fdata/1e6, np.log(ydata - d**2),
                               p0=[np.log(25), -1e-3, -1e-6])
        popt[0] = np.exp(popt[0])  # The first variable in the linear fit is ln(A_0)
    return popt, d**2


roots = '/mnt/hgfs/Shared data/BulkSynth/bulksynth-7/'  # primary energy 1e17
targetX = './fitXparameters.dat'
targetY = './fitYparameters.dat'
nameX = {}
nameY = {}
distances = [1, 4000, 7500, 11000, 15000, 37500]
for i, path in enumerate(glob.glob(os.path.join(roots, 'SIM*/'))):
    sim = path.split('/')[-2].split('_')[0]
    nameX[sim] = []
    nameY[sim] = []

    with open(os.path.join(roots, sim+'.reas'), 'r') as file:
        for line in file:
            if 'DepthOfShowerMaximum' in line:
                Xmax = float(line.split()[2])
            elif 'PrimaryParticleEnergy' in line:
                primaryE = float(line.split()[2])

    print(f"Analyzing {sim}...\n")
    t1 = time.time()
    for filename in os.listdir(path):
        temp = filename.split('_')[1].split('.')[0]
        antenna, Xslice = map(int, temp.split('x'))

        data = []
        with open(os.path.join(path, filename), 'r') as file:
            for line in file:
                data.append(line.split())
        data = np.array(data)

        freq = np.fft.rfftfreq(len(data), 2e-10)

        spectrumX = np.fft.rfft(data[:, 1])
        spectrumY = np.fft.rfft(data[:, 2])

        frange = np.logical_and(10 * 1e6 <= freq, freq <= 5 * 1e8)
        spectrumXfiltered = spectrumX[frange]
        spectrumYfiltered = spectrumY[frange]

        ampXfiltered = np.abs(spectrumXfiltered)
        ampYfiltered = np.abs(spectrumYfiltered)

        coefX, dX = amplitude_fit_slice(freq[frange], ampXfiltered, Xslice, distances[antenna])
        coefY, dY = amplitude_fit_slice(freq[frange], ampYfiltered, Xslice, distances[antenna])

        paramX = [sim, Xslice, antenna, *coefX, Xmax, primaryE]
        paramY = [sim, Xslice, antenna, *coefY, Xmax, primaryE]

        nameX[paramX[0]].append(paramX[1:])
        nameY[paramY[0]].append(paramY[1:])

        with open(targetX, 'a+') as fX, open(targetY, 'a+') as fY:
            fX.write("\t".join(map(str, paramX)))
            fX.write("\n")
            fY.write("\t".join(map(str, paramY)))
            fY.write("\n")

    t2 = time.time()
    print(f"Took {t2-t1}s to analyze\n")

# write_file_json('./fitXparameters.json', nameX)
# write_file_json('./fitYparameters.json', nameY)
