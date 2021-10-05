import numpy as np

seconds_per_year = 60 * 60 * 24 * 365.2425


def second_to_myr(seconds):
    return seconds / seconds_per_year / 1.0e6


def meter_per_second_to_cm_per_year(meter_per_second):
    return seconds_per_year * 100. * meter_per_second


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