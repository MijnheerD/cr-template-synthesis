import numpy as np
import matplotlib.pyplot as plt


PLOT = False  # whether to plot antenna shape

R = np.linspace(20.0, 200.0, 13)  # instead of 16
# R = np.concatenate((np.array([12.5 / 8, 12.5 / 4, 12.5 / 2]), R))
R_ext = np.array([225.0, 250.0, 275.0, 300.0, 350.0, 400.0, 500.0])
R = np.concatenate((R, R_ext)) * 100  # in cm!


def projected_antenna_layout(az, zen, number_of_arms=8, site=None, coreas=False):
    """

    **Properties**

    ============== ===================================================================
    Parameter      Description
    ============== ===================================================================
    *az*           arrival direction azimuth (Measured from East towards North in degrees)
    *zen*          arrival direction zenith (Zenith angle in degrees)
    ============== ===================================================================

    """
    # Convert degrees to radians and change coord system if necessary
    az = np.deg2rad(az)
    zen = np.deg2rad(zen)
    if coreas:
        az += np.pi / 2

    # Define site specific parameters
    if site == "SKA":
        inc = np.arctan(-48.27 / 27.6)  # ~ -1.05138
        altitude_cm = 46000.0
    elif site == "LOFAR":  # LOFAR site
        inc = 67.8 / 180. * np.pi
        altitude_cm = 760.0
    else:  # Template slicing technique
        inc = np.deg2rad(67.8)
        altitude_cm = 0.0

    B = np.array([0, np.cos(inc), -np.sin(inc)])  # LOFAR coordinate convention
    v = np.array([-np.cos(az) * np.sin(zen), -np.sin(az) * np.sin(zen), -np.cos(zen)])

    vxB = np.cross(v, B)
    vxB = vxB / np.linalg.norm(vxB)

    vxvxB = np.cross(v, vxB)

    # Do N=160 pattern, with intermediate positions
    radius = np.linspace(20.0, 200.0, 13)  # instead of 16
    radius_ext = np.array([225.0, 250.0, 275.0, 300.0, 350.0, 400.0, 500.0])
    radius = np.concatenate((radius, radius_ext))

    radians_step = 2 * np.pi / number_of_arms

    x = np.zeros((len(radius), number_of_arms))
    y = np.zeros((len(radius), number_of_arms))
    z = np.zeros((len(radius), number_of_arms))
    for ind, r in enumerate(radius):
        for j in np.arange(number_of_arms):
            xyz = r * (np.cos(j * radians_step) * vxB + np.sin(j * radians_step) * vxvxB)
            c = xyz[2] / v[2]

            x[ind, j] = 100 * (xyz[1] - c * v[1])
            y[ind, j] = -100 * (xyz[0] - c * v[0])
            z[ind, j] = altitude_cm

    if coreas:
        return x, y, z

    return -y, x, z


def star_shape(radius, n=8):
    """
    Space out n points on a circle with given radius.

    :param float radius: Radius of the circle
    :param int n: Number of points
    :return: Cartesian x and y coordinates of every point
    """
    angles = np.linspace(0, 2*np.pi, n+1)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)

    return x[:-1], y[:-1]


def antenna_layout(radii, n=8):
    """
    Construct star-shaped pattern of points. The coordinates are grouped in 2D matrices with the first dimension
    corresponding to the different circles and the second dimension to the point on every circle.

    The parameter n can be a list with the same length as radii, in which case the number of points on every circle can
    vary. Note that in this case the x and y matrices will contain zeros in the rows with fewer points.

    :param list radii: List of circle radii to generate.
    :param n: Number of points per circle.
    :type n: int or list
    :return: Cartesian x and y coordinates of every point
    """
    if isinstance(n, list):
        assert len(radii) == len(n), "Number of antennas per ring (param n) must have same length as number of radii"

        x_antenna = np.zeros((len(radii), max(n)))
        y_antenna = np.zeros((len(radii), max(n)))
        for i, radius in enumerate(radii):
            x_antenna[i], y_antenna[i] = star_shape(radius, n[i])

    else:
        x_antenna = np.zeros((len(radii), n))
        y_antenna = np.zeros((len(radii), n))
        for i, radius in enumerate(radii):
            x_antenna[i], y_antenna[i] = star_shape(radius, n)

    return x_antenna, y_antenna


def extract_arm(layout_x, layout_y, arm=0):
    """
    Given the Cartesian coordinates of a star-shaped pattern, extract the different arms. This assumes a constant,
    even number of points per circle. There are exactly n/2 arms in a star-shape, with n the number of points per
    circle.
    :param np.ndarray layout_x: The x-coordinates of the points
    :param np.ndarray layout_y: The y-coordinates of the points
    :param int arm: The arm to be selected, should be in [0, n/2)
    :return: Cartesian x and y coordinates of every point in the selected arm
    """
    nr_of_arms = layout_x.shape[1]

    x_select = []
    x_select.extend(layout_x[:, arm])
    x_select.extend(layout_x[:, int(nr_of_arms / 2) + arm])

    y_select = []
    y_select.extend(layout_y[:, arm])
    y_select.extend(layout_y[:, int(nr_of_arms / 2) + arm])

    return np.array(x_select), np.array(y_select)


if PLOT:
    x_r, y_r, _ = projected_antenna_layout(0, 45, coreas=True, number_of_arms=4)
    x_arm, y_arm = extract_arm(x_r, y_r, arm=1)

    print(R)
    print(f'There are {len(x_arm)} antennas in the selected arm')

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.scatter(x_r, y_r, c='grey', label='Antenna pattern')
    ax.scatter(x_arm, y_arm, c='maroon', label='Selected arm')
    ax.set_xlim([-max(R), max(R)])
    ax.set_ylim([-max(R), max(R)])

    ax.set_aspect('equal')
    ax.legend()
    plt.grid(True)

    plt.show()
