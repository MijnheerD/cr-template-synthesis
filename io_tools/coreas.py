from numpy import genfromtxt, array, sin, cos, deg2rad, dot, cross
from numpy.linalg import norm, inv
from scipy.constants import c as c_vacuum

from io_tools.tools import FileWrapper, working_directory


def norm_vector(vec):
    return vec / norm(vec)


def rot_y(theta):
    """
    Generates the matrix for a rotation around the y-axis.
    :param theta: Angle of rotation or zenith angle, in degrees
    :return: 3x3 rotation matrix
    """
    theta = deg2rad(theta)
    return array([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])


def rot_z(phi):
    """
    Generates the matrix for a rotation around the y-axis.
    :param phi: Angle of rotation or zenith angle, in degrees
    :return: 3x3 rotation matrix
    """
    phi = deg2rad(phi)
    return array([[cos(phi), -sin(phi), 0], [sin(phi), cos(phi), 0], [0, 0, 1]])


def vvB_coord_system(zenith, azimuth, b_vector=array([1, 0, 0])):
    # Normalised velocity vector
    v = array([0, 0, -1])
    v = dot(rot_y(zenith), v)
    v = dot(rot_z(azimuth), v)

    assert norm(v) - 1 < 1e-5, "Rotated v-vector not normalised!"

    # Normalised magnetic field vector
    B = norm_vector(b_vector)

    # Calculate basis vectors
    vB = norm_vector(cross(v, B))

    vvB = cross(v, vB)

    return vB, vvB, v


class CoreasFileWrapper(FileWrapper):
    """
    Wrapper to access and interpret data from raw CoREAS files. Allows to read in the traces and transform them to
    different coordinate systems.
    """

    def __init__(self, fpath):
        super().__init__(fpath, 'r')

        self.coord_system = 'xyz'

        self.read()

    # Reimplement read/write functions
    def read(self):
        print(f"Moving to {self.dir}")
        with working_directory(self.dir):
            # conversion from statV/cm to microV/m
            print(f"Reading from {self.file}")
            self.contents = genfromtxt(self.file) * c_vacuum * 1e2

    def write(self):
        raise PermissionError("Cannot write to CoREAS raw file")

    # CoREAS specific functions
    # Content getters, based on CoREAS file content
    def get_time(self):
        return self.contents[:, 0]

    def get_traces(self):
        return self.contents[:, 1:]

    # Data manipulation functions
    def set_coord_system(self, system, zenith=None, azimuth=None, site='LOFAR', B=None):
        """
        Change the traces to be in a new coordinate system. Note that the angles are defined as in CORSIKA/CoREAS.
        :param system: String identifying the coordinate system. Must be either "xyz" or "vvB".
        :param zenith: Zenith angle (measured from the z-axis down) in degrees.
        :param azimuth: Azimuth angle (measured counterclock wise from the x-axis) in degrees.
        :param site: Set a site with preloaded magnetic field vector.
        :param B: Custom magnetic field vector to use. Is ignored when valid site is used.
        :return:
        """
        # Check whether change is necessary
        if system == self.coord_system:
            print(f"Coordinate system already set to {system}")
            return

        # Set the magnetic field vector
        if site == 'LOFAR':
            print("Using LOFAR magnetic field vector")
            B_vector = norm_vector(array([18.6, 0, -45.6]))
        else:
            assert B is not None, "Site not recognized, please provide magnetic field vector explicitly"

            B_vector = norm_vector(B)

        # Build transformation matrix
        assert zenith is not None and azimuth is not None, "Need zenith and azimuth angle to convert traces"

        if system == 'vvB':
            self.coord_system = 'vvB'

            # Get basis vectors (matrices with dimensions 3x1)
            vB, vvB, v = vvB_coord_system(zenith, azimuth, B_vector)

            trans_matrix = array([vB, vvB, v]).T

        elif system == 'xyz':
            self.coord_system = 'xyz'

            # Get basis vectors (matrices with dimensions 3x1)
            vB, vvB, v = vvB_coord_system(zenith, azimuth, B_vector)

            # Create transformation matrix and take the inverse
            trans_matrix = array([vB, vvB, v]).T
            trans_matrix = inv(trans_matrix)

        else:
            raise ValueError("Coordinate system not recognized")

        # Apply transformation
        E = self.get_traces()  # matrix with dimensions nx3

        # Transform traces to new coordinate system
        self.contents[:, 1:] = dot(E, trans_matrix)
