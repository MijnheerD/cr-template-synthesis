import os
import numpy as np
import pandas as pd
from NumpyEncoder import read_file_json
from GlobalVars import DISTANCES, DATABASE_DIRECTORY, FIT_DIRECTORY, REAS_DIRECTORY
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV


def read_long(file):
    long = np.genfromtxt(file, skip_header=2, skip_footer=216, usecols=(0, 2, 3))
    energy = np.genfromtxt(file, skip_header=212, skip_footer=6, usecols=(0, 1, 2, 3))
    return long[:, 0], np.sum(long[:, 1:], axis=1), np.sum(energy[:, 1:], axis=1)


features = ['Xmax', 'Nmax', 'L', 'R', 'X1', 'mass', 'fraction', 'DeltaX', 'antenna']
# features = ['Xmax', 'Xslice', 'antenna']
targets = ['A', 'b', 'c']


database = read_file_json(os.path.join(DATABASE_DIRECTORY, 'sim_parameters.json'))
database = pd.DataFrame(database, columns=['sim', 'Xmax', 'Nmax', 'L', 'R', 'X1', 'mass'])
database = database.set_index('sim')

FIT_FILES = ['fitFilesUB17', 'fitFilesUB18', 'fitFilesUB19']
FIT_DIRS = [os.path.join(FIT_DIRECTORY, file) for file in FIT_FILES]

df_list = []
for lib in FIT_DIRS:
    for pol in ['fitX', 'fitY']:
        for file in os.listdir(os.path.join(lib, pol)):
            sim_nr = file.split('.')[0][3:]
            x_slices, n_slices, e_slices = read_long(os.path.join(REAS_DIRECTORY, f'DAT{sim_nr}.long'))

            frame = pd.read_csv(os.path.join(lib, pol, file), sep=', ', engine='python',
                                names=['Xslice', 'Antenna', 'A', 'b', 'c', 'Xmax', 'E'],
                                dtype={'Xslice': 'uint32', 'Antenna': 'uint32', 'A': 'float64', 'b': 'float64',
                                       'c': 'float64', 'Xmax': 'float32', 'E': 'float64'})
            frame.sort_values(['Antenna', 'Xslice'], inplace=True)

            frame = frame.assign(Nslice=list(n_slices) * 6)
            frame = frame.assign(EM=list(e_slices) * 6)
            frame = frame.assign(fraction=lambda x: np.sum(e_slices) / x.E * 1e9)
            frame = frame.assign(Polarization=0) if pol == 'fitX' else frame.assign(Polarization=1)

            series = database.loc[sim_nr]
            data = pd.DataFrame([series]*len(frame))

            del frame['Xmax']  # Avoid double Xmax column, use the one from R-L fit
            df_list.append(pd.concat([frame, data.reset_index(drop=True)], axis=1))

df = pd.concat(df_list)
df['antenna'] = df.apply(lambda x: DISTANCES[int(x['Antenna'])], axis=1)
df['DeltaX'] = df['Xslice'] - df['Xmax']

grid = {
    'n_estimators': [50, 100, 200, 400],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [3, 5, 10, 25],
    'random_state': [42]
}

cv_results = []
for target in targets:
    X = df.loc[:, features].values
    y = df.loc[:, target].values

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, train_size=0.80, test_size=0.20, random_state=4
    # )

    CV_rfr = GridSearchCV(estimator=RandomForestRegressor(), param_grid=grid, cv=5, n_jobs=2, verbose=4)
    CV_rfr.fit(X, y)

    cv_results.append(pd.DataFrame(CV_rfr.cv_results_))
