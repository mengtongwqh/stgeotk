import math
import numpy as np


def line_to_cartesian(tpdata, lower_hemisphere=True):
    """
    Convert lineation trend/plunge data of
    to directional cosines in Cartesian coordinates.
    Assuming lower-hemisphere.
    """
    if __debug__:
        assert np.all(np.abs(tpdata.T[1]) <= 90.0)

    trend, plunge = np.radians(tpdata).T
    x = np.cos(plunge) * np.sin(trend)
    y = np.cos(plunge) * np.cos(trend)
    z = -np.sin(plunge) if lower_hemisphere else np.sin(plunge)
    return np.array([x, y, z]).T


def cartesian_to_line(xyz):
    """
    Convert Cartesian coordinates to trend and plunge
    """
    x, y, z = xyz.T
    x2y2 = x*x + y*y
    trend = (np.arctan2(x, y) + math.pi * (z > 0.0)) % (2 * math.pi)
    plunge = np.abs(np.abs(np.arctan2(z, np.sqrt(x2y2))))
    return np.degrees(np.array([trend, plunge]).T)


def line_to_spherical(tpdata):
    """
    Convert lineation trend/plunge data (in degrees)
    to spherical angles (in radians)
    """
    trend, plunge = np.radians(tpdata).T
    theta, phi = (90.0-trend) % 360.0, 90.0 + plunge
    return np.radians(np.array([theta, phi])).T


def spherical_to_line(theta_phi):
    """
    Convert the spherical coordinates
    to geological trend and plunge
    also convert from radians to degree
    Not yet tested
    """
    theta, phi = theta_phi[:, 1], theta_phi[:, 2]
    trend = np.degrees(0.5 * math.pi + math.pi *
                      (np.degrees(phi) > 0.0) - theta) % 360.0
    plunge = np.abs(np.degrees(phi) - 90.0)
    return np.array([trend, plunge]).T


def spherical_to_cartesian(theta_phi_radius):
    if theta_phi_radius.shape[1] == 2:
        theta, phi = theta_phi_radius
        radius = 1.0
    elif theta_phi_radius.shape[1] == 3:
        theta, phi, radius = theta_phi_radius
    else:
        raise RuntimeError(
            "Unexpected input dimension for spherical coordinate")

    x = radius*np.sin(phi)*np.cos(theta)
    y = radius*np.sin(phi)*np.sin(theta)
    z = radius*np.cos(phi)
    return np.array([x, y, z]).T


def cartesian_to_spherical(xyz):
    """
    Convert a vector in Cartesian coordinates and
    return theta, rho (in radians) and the magnitude
    -----------
    Parameters:
    -----------
    xyz: data points in N x 3 array
    """
    x, y, z = xyz.T
    x2y2 = x*x + y*y

    theta = np.arctan2(y, x)
    phi = np.arctan2(np.sqrt(x2y2), z)
    radius = np.sqrt(x2y2 + z*z)
    return np.array([theta, phi, radius]).T


def rotation_matrix(axis, rotation_degree):
    """
    Compute a rotation matrix around a coordinate axis
    -----------
    Parameters:
    -----------
    axis (character 'x', 'y' or 'z'):
        coordinate axis around which the dataset is rotated
    angle (in degree):
        rotation angle in the positive
        direction of the axis, i.e. counterclockwise
    """
    # construct the rotation matrix
    rotmat = np.zeros((3, 3), dtype=np.double)
    angle = math.radians(rotation_degree)
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)

    if axis == 'z':  # rotation in xy-plane
        rotmat[2, 2] = 1.0
        rotmat[0, 0] = rotmat[1, 1] = cos_angle
        rotmat[0, 1] = -sin_angle
        rotmat[1, 0] = sin_angle
    elif axis == 'x':  # rotation in yz-plane
        rotmat[0, 0] = 1.0
        rotmat[1, 1] = rotmat[2, 2] = cos_angle
        rotmat[1, 2] = -sin_angle
        rotmat[2, 1] = sin_angle
    elif axis == 'y':  # rotation in the xz-plane
        rotmat[1, 1] = 1.0
        rotmat[0, 0] = rotmat[2, 2] = cos_angle
        rotmat[0, 2] = sin_angle
        rotmat[2, 0] = -sin_angle
    else:
        raise RuntimeError(f"Unexpected coordinate axis {axis}")

    return rotmat


def plane_nodes(strike, dip, n_segments):
    """
    Generate nodes of a plane of given strike and dip
    """
    n_nodes = n_segments + 1
    stk_rot = rotation_matrix('z', -strike)
    dip_rot = rotation_matrix('y', dip)
    rotmat = stk_rot.dot(dip_rot)

    nodes_trd = np.linspace(0, math.pi, n_nodes)  # trend angle
    nodes = np.vstack(
        (np.sin(nodes_trd), np.cos(nodes_trd), np.zeros(n_nodes)))
    return rotmat.dot(nodes).T


def pole_to_plane(strike_dip):
    """
    return the pole to the plane,
    in the format of trend and plunge.
    """
    if isinstance(strike_dip, np.ndarray):
        strike, dip = strike_dip.T
        normal_trd = (strike - 90.0) % 360.0
        normal_plg = 90.0 - dip
        return  np.array([normal_trd, normal_plg]).T
    elif isinstance(strike_dip, list):
        return [(strike_dip[0] - 90.0) % 360.0, 90.0 - strike_dip[1]]
    else:
        raise NotImplementedError()


def plane_from_pole(pole_tp):
    """
    return the plane attitude in strike-dip "line" format
    from the given pole data
    """
    if isinstance(pole_tp, np.ndarray):
        trend, plunge = pole_tp.T
        plane_stk = (trend + 90.0) % 360.0
        plane_dip = 90.0 - plunge
        return  np.array([plane_stk, plane_dip]).T
    elif isinstance(pole_tp, list):
        return [(pole_tp[0] + 90.0) % 360.0, 90.0 - pole_tp[1]]
    else:
        raise NotImplementedError()

