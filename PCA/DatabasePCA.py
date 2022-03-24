import os
import mpl_axes_aligner
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from NumpyEncoder import read_file_json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from GlobalVars import DATABASE_DIRECTORY, FIT_DIRECTORY, REAS_DIRECTORY


def read_long(file, source=REAS_DIRECTORY):
    if source is not None:
        file = os.path.join(REAS_DIRECTORY, f'DAT{file}.long')
    long = np.genfromtxt(file, skip_header=2, skip_footer=216, usecols=(0, 2, 3))
    energy = np.genfromtxt(file, skip_header=212, skip_footer=6, usecols=(0, 1, 2, 3))
    return long[:, 0], np.sum(long[:, 1:], axis=1), np.sum(energy[:, 1:], axis=1)


features = ['Xmax', 'Nmax', 'L', 'R', 'X1', 'mass', 'fraction']
pca = PCA(n_components=2)

database = read_file_json(os.path.join(DATABASE_DIRECTORY, 'sim_parameters.json'))
database = pd.DataFrame(database, columns=['sim', 'Xmax', 'Nmax', 'L', 'R', 'X1', 'mass'])
database = database.set_index('sim')

database['type'] = database.apply(lambda row: row.name[0], axis=1)
database['E'] = database.apply(lambda row: int(row.name[1]) + 17, axis=1)
database['EM'] = database.apply(lambda row: np.sum(read_long(row.name)[2]), axis=1)

database['fraction'] = database['EM'] / (np.power(10, database['E'] - 9))

p17 = database.loc[((database['E'] == 17) & (database['type'] == '1')), features].values
p18 = database.loc[((database['E'] == 18) & (database['type'] == '1')), features].values
p19 = database.loc[((database['E'] == 19) & (database['type'] == '1')), features].values
f17 = database.loc[((database['E'] == 17) & (database['type'] == '2')), features].values
f18 = database.loc[((database['E'] == 18) & (database['type'] == '2')), features].values
f19 = database.loc[((database['E'] == 19) & (database['type'] == '2')), features].values

data = [p17, p18, p19, f17, f18, f19]
names = ['Proton, 1e17 eV', 'Proton, 1e18 eV', 'Proton, 1e19 eV', 'Iron, 1e17 eV', 'Iron, 1e18 eV', 'Iron, 1e19 eV']
norm = [StandardScaler().fit_transform(x) for x in data]

frames = []
loadings = []
for series in norm:
    principal = pca.fit_transform(series)
    frames.append(pd.DataFrame(data=principal, columns=['Component 1', 'Component 2']))
    loadings.append(pd.DataFrame(data=pca.components_, columns=features))

fig1, ax1 = plt.subplots(2, 3, num=1, figsize=(25, 12))
ax1 = ax1.flatten()
ax2 = [ax.twinx().twiny() for ax in ax1]

for ind, frame in enumerate(frames):
    sns.scatterplot(data=frame, x='Component 1', y='Component 2', ax=ax1[ind])
    ax1[ind].set_title(names[ind])
    # align x = 0 of ax and ax2 with the center of figure
    mpl_axes_aligner.align.xaxes(ax1[ind], 0, ax2[ind], 0, 0.5)
    # align y = 0 of ax and ax2 with the center of figure
    mpl_axes_aligner.align.yaxes(ax1[ind], 0, ax2[ind], 0, 0.5)

    for feature in features:
        ax2[ind].arrow(0, 0, loadings[ind].loc[0, feature], loadings[ind].loc[1, feature], alpha=0.5)
        ax2[ind].text(loadings[ind].loc[0, feature] * 1.05, loadings[ind].loc[1, feature] * 1.05, feature,
                      ha='center', va='center')

plt.savefig('../Figures/PCA_Etype.png', bbox_inches='tight')
