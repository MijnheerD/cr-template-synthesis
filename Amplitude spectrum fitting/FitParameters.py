import matplotlib.pyplot as plt

distances = [1, 4000, 7500, 11000, 15000, 37500]

paramXfile = './fitXparameters.dat'
paramYfile = './fitYparameters.dat'
Xslice = 655
antenna = 3

A_0_x = []
b_x = []
c_x = []
X_max_x = []

A_0_y = []
b_y = []
c_y = []
X_max_y = []

with open(paramXfile, 'r') as fX, open(paramYfile, 'r') as fY:
    for lineX, lineY in zip(fX, fY):
        lstX = lineX.split()
        lstY = lineY.split()
        if lstX[1] == str(Xslice):
            if lstX[2] == str(antenna):
                A_0_x.append(float(lstX[3]))
                b_x.append(float(lstX[4]))
                c_x.append(float(lstX[5]))
                X_max_x.append(float(lstX[6]))
                A_0_y.append(float(lstY[3]))
                b_y.append(float(lstY[4]))
                c_y.append(float(lstY[5]))
                X_max_y.append(float(lstY[6]))

fig1, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 6))

ax1.scatter(X_max_x[:100], A_0_x[:100], color='cyan')
ax1.scatter(X_max_x[100:], A_0_x[100:], color='cyan', marker='x')
ax2.scatter(X_max_y[:100], A_0_y[:100], color='cyan')
ax2.scatter(X_max_y[100:], A_0_y[100:], color='cyan', marker='x')

ax1.set_xlabel(r"$X_{max}[g/cm^2]$")
ax1.set_ylabel(r"$A_0$ (x-component) [a.u.]")
ax1.set_title(f"X = {Xslice} g/cm^2 r = {distances[antenna] / 100} m")
ax1.ticklabel_format(axis='y', useMathText=True, scilimits=(0, 0))

ax2.set_xlabel(r"$X_{max}[g/cm^2]$")
ax2.set_ylabel(r"$A_0$ (y-component) [a.u.]")
ax2.set_title(f"X = {Xslice} g/cm^2 r = {distances[antenna] / 100} m")
ax2.ticklabel_format(axis='y', useMathText=True, scilimits=(0, 0))

plt.savefig(f'A0_{antenna}x{Xslice}_1e17.png', bbox_inches='tight')

fig2, [ax3, ax4] = plt.subplots(1, 2, figsize=(12, 6))

ax3.scatter(X_max_x[:100], b_x[:100], color='cyan')
ax3.scatter(X_max_x[100:], b_x[100:], color='cyan', marker='x')
ax4.scatter(X_max_y[:100], b_y[:100], color='cyan')
ax4.scatter(X_max_y[100:], b_y[100:], color='cyan', marker='x')

ax3.set_xlabel(r"$X_{max}[g/cm^2]$")
ax3.set_ylabel(r"$b$ (x-component) [1/MHz]")
ax3.set_title(f"X = {Xslice} g/cm^2 r = {distances[antenna] / 100} m")
ax3.ticklabel_format(axis='y', useMathText=True, scilimits=(0, 0))

ax4.set_xlabel(r"$X_{max}[g/cm^2]$")
ax4.set_ylabel(r"$b$ (y-component) [1/MHz]")
ax4.set_title(f"X = {Xslice} g/cm^2 r = {distances[antenna] / 100} m")
ax4.ticklabel_format(axis='y', useMathText=True, scilimits=(0, 0))

plt.savefig(f'b_{antenna}x{Xslice}_1e17.png', bbox_inches='tight')

fig3, [ax5, ax6] = plt.subplots(1, 2, figsize=(12, 6))

ax5.scatter(X_max_x[:100], c_x[:100], color='cyan')
ax5.scatter(X_max_x[100:], c_x[100:], color='cyan', marker='x')
ax6.scatter(X_max_y[:100], c_y[:100], color='cyan')
ax6.scatter(X_max_y[100:], c_y[100:], color='cyan', marker='x')

ax5.set_xlabel(r"$X_{max}[g/cm^2]$")
ax5.set_ylabel(r"$c$ (x-component) [$1/MHz^2$]")
ax5.set_title(f"X = {Xslice} g/cm^2 r = {distances[antenna] / 100} m")
ax5.ticklabel_format(axis='y', useMathText=True, scilimits=(0, 0))

ax6.set_xlabel(r"$X_{max}[g/cm^2]$")
ax6.set_ylabel(r"$c$ (y-component) [$1/MHz^2$]")
ax6.set_title(f"X = {Xslice} g/cm^2 r = {distances[antenna] / 100} m")
ax6.ticklabel_format(axis='y', useMathText=True, scilimits=(0, 0))

plt.savefig(f'c_{antenna}x{Xslice}_1e17.png', bbox_inches='tight')