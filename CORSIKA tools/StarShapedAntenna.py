import numpy as np
import matplotlib.pyplot as plt


PLOT = False  # whether to plot antenna shape

R = np.linspace(20.0, 200.0, 13)  # instead of 16
# R = np.concatenate((np.array([12.5 / 8, 12.5 / 4, 12.5 / 2]), R))
R_ext = np.array([225.0, 250.0, 275.0, 300.0, 350.0, 400.0, 500.0])
R = np.concatenate((R, R_ext))


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
    x_r, y_r = antenna_layout(R)
    x_arm, y_arm = extract_arm(x_r, y_r, arm=0)

    print(R)
    print(f'There are {len(x_arm)} antennas in the selected arm')

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.scatter(x_r, y_r, c='grey', label='Antenna pattern')
    ax.scatter(x_arm, y_arm, c='maroon', label='Selected arm')
    ax.set_xlim([-max(R), max(R)])
    ax.set_ylim([-max(R), max(R)])

    plt.grid(True)
    ax.set_aspect('equal')

    plt.show()
