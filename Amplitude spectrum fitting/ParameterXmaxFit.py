import os
from scipy.optimize import curve_fit


FIT_DIRECTORY = 'fitFiles7/'  # location of the files containing the fit parameters
PARAM_DIRECTORY = 'paramFit7/'  # location to store the parameters of the quadratic fit
DISTANCES = [1, 4000, 7500, 11000, 15000, 37500]  # antenna radial distances to shower core, in cm
XSLICE = 550
ANTENNA = 3


X_max = []

filesX = []  # list of file handles for fitX files
filesY = []  # list of file handles for fitY files
for file in os.listdir(os.path.join(FIT_DIRECTORY, 'fitX/')):
    filesX.append(open(os.path.join(FIT_DIRECTORY, 'fitX/', file)))
    filesY.append(open(os.path.join(FIT_DIRECTORY, 'fitY/', file)))

reading = True  # Loop over all slices, aka every line in every file
X_maxFilled = False  # Read Xmax data only once, as it stays the same for every iteration
while reading:
    # Reset parameter lists for X
    A_0_x = []
    b_x = []
    c_x = []
    # Reset parameter lists for Y
    A_0_y = []
    b_y = []
    c_y = []
    # Loop over all the files in polarization pairs
    for (fileX, fileY) in zip(filesX, filesY):
        lstX = fileX.readline().split(', ')
        lstY = fileY.readline().split(', ')
        # Check if end of file is reached
        if len(lstX) == 1:
            reading = False
            break

        if (lstX[0] != lstY[0]) or (lstX[1] != lstY[1]):
            reading = False
            print('Not working on the same line', fileX.name, fileY.name, sep='\n')
            break

        if not X_maxFilled:
            assert lstX[5] == lstY[5], f"X and Y have different Xmax, {lstX[5]} and {lstY[5]}"
            X_max.append(float(lstX[5]))

        A_0_x.append(float(lstX[2]))
        b_x.append(float(lstX[3]))
        c_x.append(float(lstX[4]))
        A_0_y.append(float(lstY[2]))
        b_y.append(float(lstY[3]))
        c_y.append(float(lstY[4]))

    if not reading:
        break

    # Perform a quadratic fit to every parameter
    resX_A, _ = curve_fit(lambda x, p0, p1, p2: p0 + p1 * x + p2 * x ** 2, X_max, A_0_x)
    resX_b, _ = curve_fit(lambda x, p0, p1, p2: p0 + p1 * x + p2 * x ** 2, X_max, b_x)
    resX_c, _ = curve_fit(lambda x, p0, p1, p2: p0 + p1 * x + p2 * x ** 2, X_max, c_x)
    resY_A, _ = curve_fit(lambda x, p0, p1, p2: p0 + p1 * x + p2 * x ** 2, X_max, A_0_y)
    resY_b, _ = curve_fit(lambda x, p0, p1, p2: p0 + p1 * x + p2 * x ** 2, X_max, b_y)
    resY_c, _ = curve_fit(lambda x, p0, p1, p2: p0 + p1 * x + p2 * x ** 2, X_max, c_y)

    # Save fitted parameters to file
    with open(os.path.join(PARAM_DIRECTORY, 'fitX', 'slice' + lstX[0] + '.dat'), 'a+') as f:
        f.write('antenna' + lstX[1] + '\t')
        f.writelines(map(lambda x: str(x) + '\t', resX_A))
        f.writelines(map(lambda x: str(x) + '\t', resX_b))
        f.writelines(map(lambda x: str(x) + '\t', resX_c))
        f.write('\n')
    with open(os.path.join(PARAM_DIRECTORY, 'fitY', 'slice' + lstY[0] + '.dat'), 'a+') as f:
        f.write('antenna' + lstY[1] + '\t')
        f.writelines(map(lambda x: str(x) + '\t', resY_A))
        f.writelines(map(lambda x: str(x) + '\t', resY_b))
        f.writelines(map(lambda x: str(x) + '\t', resY_c))
        f.write('\n')

    X_maxFilled = True

# Close all the files once done
for (fileX, fileY) in zip(filesX, filesY):
    fileX.close()
    fileY.close()
