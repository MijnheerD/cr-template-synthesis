import os
import glob
import numpy as np
from NumpyEncoder import write_file_json
from SliceFit import amplitude_fit_slice, distances


roots = '/mnt/hgfs/Shared data/BulkSynth/bulksynth-7/'  # primary energy 1e17
targetX = './fitXparameters.dat'
targetY = './fitYparameters.dat'
nameX = {}
nameY = {}
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

    print(f"Analyzing {sim}...")
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

write_file_json('./fitXparameters.json', nameX)
write_file_json('./fitYparameters.json', nameY)
