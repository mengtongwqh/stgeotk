import math

import numpy as np

from .dataset import DatasetBase, LineData
from .utility import line_to_cartesian


class CountingGrid:
    def __init__(self, grid_type):
        self._grid_type = grid_type

    def generate(self, degree_spacing=2.5):
        if self._grid_type == "hemispherical":
            return self._hemispherical_grid(degree_spacing)
        elif self._grid_type == "spherical":
            return self._spherical_grid(degree_spacing)
        else:
            raise RuntimeError(
                f"Unknown counting grid type {self._grid_types}")


    def _hemispherical_grid(self, degree_spacing):
        '''
        Construct the counting grid in the lineation coordinate
        Assuming lower hemispheric
        '''
        spacing = math.radians(degree_spacing)
        nodes = [(0.0, 90.0)]  # south pole
        # equator
        for theta in np.arange(0., 360.,  degree_spacing):
            nodes.append((theta + 90.0 + degree_spacing / 2.0, 0.0))
        # ordinary points on sphere
        for phi in np.arange(degree_spacing, 90.0, degree_spacing):
            azm_spacing = math.sin(spacing/2.0) / math.sin(math.radians(phi))
            azm_spacing = math.degrees(2*math.asin(azm_spacing))
            for theta in np.arange(0.0, 360.0, azm_spacing):
                nodes.append((theta + phi + azm_spacing/2.0, 90.0 - phi))
        # convert to directional cosines
        return line_to_cartesian(np.array(nodes))

    def _spherical_grid(self, degree_spacing):
        '''
        Construct the counting grid for the whole sphere
        '''
        spacing = math.radians(degree_spacing)
        nodes = [(0.0, 90.0), (0.0, -90.0)]  # south and north pole
        # equator
        for theta in np.arange(0., 360.,  degree_spacing):
            nodes.append((theta + 90.0 + degree_spacing / 2.0, 0.0))
        # ordinary points on sphere
        for phi in np.arange(degree_spacing, 90.0, degree_spacing):
            azm_spacing = math.sin(spacing/2.0) / math.sin(math.radians(phi))
            azm_spacing = math.degrees(2*math.asin(azm_spacing))
            for theta in np.arange(0.0, 360.0, azm_spacing):
                nodes.append((theta + phi + azm_spacing/2.0, 90.0 - phi))
                nodes.append((theta + phi + azm_spacing/2.0, phi - 90.0))
        return line_to_cartesian(np.array(nodes))


class ContourData(DatasetBase):
    '''
    Contour data
    '''
    def __init__(self, grid_spacing_degree=2.5, **kwargs):
        # counting grid
        super().__init__(3)
        self._grid_spacing = grid_spacing_degree


    def _set_data(self, value):
        if isinstance(value, LineData):
            self._data = value
            grid = CountingGrid("hemispherical")
            self._nodes = grid.generate(self._grid_spacing)
            # self._data_legend = value.legend() + "Density"
        else:
            raise RuntimeError(f"{type(value)} cannot be contoured.")

    @property
    def nodes(self):
        return self._nodes

    def optimize_k(self):
        '''
        Optimizes the value of K from the data, 
        using Diggle and Fisher (88) method.
        '''
        from scipy.optimize import minimize_scalar
        datac = self.data_to_contour.data
        # objective function to be minimized

        def obj(k):
            W = np.exp(k*(np.abs(np.dot(datac, datac.T))))\
                * (k/(4*np.pi*np.sinh(k+1e-9)))
            np.fill_diagonal(W, 0.)
            return -np.log(W.sum(axis=0)).sum()
        return minimize_scalar(obj).x

    def count_kamb(self, counting_angle=None):
        '''
        Performs counting as in Robin and Jowett (1986), 
        which based on Kamb (1956). 
        Will estimate an appropriate counting angle if not give.
        '''
        if counting_angle is None:
            n = self.data.n_entries
            theta = (n - 1.0) / (n + 1.0)
        else:
            theta = math.cos(math.radians(counting_angle))
        datac = self.data.data
        return np.where(np.abs(np.dot(self.nodes, datac.T))
                        >= theta, 1, 0).sum(axis=1)
