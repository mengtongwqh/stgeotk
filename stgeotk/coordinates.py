
import math
import numpy as np

def line_to_cartesian(tpdata, lower_hemisphere=True):
    '''
    Convert lineation trend/plunge data of 
    to directional cosines in Cartesian coordinates.
    Assuming lower-hemisphere 
    '''
    if __debug__:
        assert(np.all(np.abs(tpdata.T[1]) <= 90.0))

    trend, plunge = np.radians(tpdata).T
    x = np.cos(plunge) * np.sin(trend)
    y = np.cos(plunge) * np.cos(trend)
    z = -np.sin(plunge) if lower_hemisphere else np.sin(plunge)
    return np.array([x, y, z]).T


def line_to_spherical(tpdata):
    '''
    Convert lineation trend/plunge data (in degrees)  
    to spherical angles (in radians)
    '''
    trend, plunge = np.radians(tpdata).T
    theta, phi = (90.0-trend) % 360.0, 90.0 + plunge
    return np.radians(np.array([theta, phi])).T


def spherical_to_cartesian(theta_phi_radius):
    if theta_phi_radius.shape[1] == 2:
        theta, phi = theta_phi_radius
        radius = 1.0
    elif theta_phi_radius.shape[1] == 3:
        theta, phi, radius = theta_phi_radius
    else:
        raise RuntimeError(
            "Unexpected input dimension for spherical coordinate")

    x = radius*np.sin(phi)*np.cos(theta),
    y = radius*np.sin(phi)*np.sin(theta),
    z = radius*np.cos(phi)
    return np.array([x, y, z]).T


def spherical_to_line(theta_phi):
    '''
    Convert the spherical coordinates 
    to geological trend and plunge
    also convert from radians to degree
    Not yet tested
    '''
    theta, phi = theta_phi[:, 1], theta_phi[:, 2]
    trend = np.degree(0.5 * math.pi + math.pi *
                      (np.degree(phi) > 0.0) - theta) % 360.0
    plunge = np.abs(np.degree(phi) - 90.0)
    return np.array([trend, plunge]).T


def cartesian_to_line(xyz):
    '''
    Convert Cartesian coordinates to trend and plunge
    '''
    x, y, z = xyz.T
    x2y2 = x*x + y*y
    trend = (np.arctan2(x, y) + math.pi * (z > 0.0)) % (2 * math.pi)
    plunge = np.abs(np.abs(np.arctan2(z, np.sqrt(x2y2))))
    return np.degrees(np.array([trend, plunge]).T)


def cartesian_to_spherical(xyz):
    '''
    Convert a vector in Cartesian coordinates and
    return theta, rho (in radians) and the magnitude
    '''
    x, y, z = xyz.T
    x2y2 = x*x + y*y

    theta = np.arctan2(y, x)
    phi = np.arctan2(np.sqrt(x2y2), z)
    radius = np.sqrt(x2y2 + z*z)
    return np.array([theta, phi, radius]).T
