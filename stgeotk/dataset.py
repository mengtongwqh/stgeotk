import numpy as np
import math
from . utility import line_to_cartesian, deref_or_default, logger
from abc import ABC, abstractmethod

import warnings


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
                "Unknown counting grid type {0}".format(self._grid_types))

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


class DatasetBase(ABC):
    def __init__(self, spacedim):
        self._spacedim = spacedim
        self._data = None
        self._data_legend = ""
        self._color_data = None
        self._color_legend = ""
        self._n_entries = 0

    @property
    def spacedim(self):
        return self._spacedim

    @property
    def n_entries(self):
        return self._n_entries

    @property
    def data(self):
        return self._data

    @property
    def data_legend(self):
        return self._data_legend

    @property
    def color_data(self):
        return self._color_data

    @property
    def color_legend(self):
        return self._color_legend

    @data.setter
    def data(self, value):
        self._set_data(value)

    @color_data.setter
    def color_data(self, value):
        if len(value) != self.n_entries:
            raise RuntimeError("The dataset has length {0} but the color data has length {1}".format(
                self.n_entries, len(value)))
        self._color_data = value

    @abstractmethod
    def _set_data(self, value):
        pass


class LineData(DatasetBase):
    '''
    Stores linear structural data (or lineation),
    in the form of directional cosines.
    '''

    def __init__(self, **kwargs):
        super().__init__(3)
        # lower/upper hemispheric projection
        self._lower_hemisphere = True
        if "lower_hemisphere" in kwargs:
            self._lower_hemisphere = kwargs["lower_hemisphere"]

    @property
    def color_data(self):
        return self._color_data

    @color_data.setter
    def color_data(self, value):
        if value is None:
            return
        if len(value) != self._n_entries:
            raise RuntimeError(
                "Color data must have same number of entries as the original data: {0} != {1}".
                format(len(value), self.n_entries))
        self._color_data = value

    def load_data(self,
                  data, data_legend="Lineation",
                  color_data=None, color_legend=""):
        '''
        Load the dataset
        '''
        self.data = data
        self.color_data = color_data
        self._data_legend = data_legend
        self._color_legend = color_legend
        logger.info("LineData of {0} entries are loaded with legend \"{1}\"".
                    format(self.n_entries, self.data_legend))

    def _set_data(self, value):
        '''
        Parse the dataset entries.
        The entries can either be in the vector form of n_entries x 3
        or in the trend/plunge form of n_entries x 2 (in degree)
        '''
        if value.shape[1] == 2:  # trend/plunge format
            self._data = line_to_cartesian(value)
        elif value.shape[1] == 3:
            # normalize to unit vector
            self._data = value / np.linalg.norm(value, axis=1)[:, np.newaxis]
            self.__restrict_to_hemisphere(self._lower_hemisphere)
        else:
            raise(RuntimeError(
                "Unexpected dimension for input dataset: {0}".format(value.shape)))
        self._n_entries = self.data.shape[0]

    def __restrict_to_hemisphere(self, lower_hemisphere):
        '''
        Restrict non-directional dataset to the upper/lower hemisphere
        '''
        if __debug__ and self._spacedim != 3:
            raise RuntimeError(
                "Hemisphere restriction only makes sense for 3d dataset.")
        if lower_hemisphere:
            upper_hemis = (self._data[:, 2] > 0)
            self._data[upper_hemis, :] = -self._data[upper_hemis, :]
        else:
            lower_hemis = self._data[:, 2] < 0
            self._data[lower_hemis, :] = -self._data[lower_hemis, :]


class ContourData(DatasetBase):
    '''
    Object for computing the density contour.
    '''

    def __init__(self, data_to_contour=None, **kwargs):
        # counting grid
        super().__init__(3)
        self._grid_spacing = deref_or_default(kwargs, "grid_spacing", 2.5)
        self._counting_method = deref_or_default(kwargs, "method", "kamb")
        self._counting_angle = deref_or_default(kwargs, "counting_angle", None)
        self._color_data = None
        if data_to_contour is not None:
            self.load_data(data_to_contour)

    def load_data(self, data_to_contour):
        if isinstance(data_to_contour, LineData):
            self._dataset_to_contour = data_to_contour
            self._data_legend = data_to_contour.data_legend + "CountingGrid"
            # instantiate a counting grid
            grid = CountingGrid("hemispherical")
            self._data = grid.generate(self._grid_spacing)
            self._n_entries = self._data.shape[0]
            self.count()
            self._color_legend = data_to_contour.data_legend + " density (%)"
        else:
            raise RuntimeError(
                "Dataset type {0} cannot be contoured.".format(type(data_to_contour)))

    @ property
    def dataset_to_contour(self):
        return self._dataset_to_contour

    @ property
    def nodes(self):
        return self._data

    def count(self):
        if self._counting_method == "kamb":
            count = self._count_kamb(self._counting_angle)
            logger.info("ContourData count [{0}, {1}]".format(
                min(count), max(count)))
            return count
        else:
            raise RuntimeError(
                "Unknown counting method {0}".format(self._counting_method))

    def _set_data(self, value):
        raise RuntimeError(
            "data attributes in ContourData is set automatically set. Do not use this setter.")

    def _optimize_k(self):
        '''
        Optimizes the value of K from the data, 
        using Diggle and Fisher (88) method.
        '''
        from scipy.optimize import minimize_scalar
        datac = self.dataset_to_contour.data
        # objective function to be minimized

        def obj(k):
            W = np.exp(k*(np.abs(np.dot(datac, datac.T))))\
                * (k/(4*np.pi*np.sinh(k+1e-9)))
            np.fill_diagonal(W, 0.)
            return -np.log(W.sum(axis=0)).sum()
        return minimize_scalar(obj).x

    def _count_kamb(self, counting_angle):
        '''
        Performs counting as in Robin and Jowett (1986), 
        which is based on Kamb (1956). 
        Will estimate an appropriate counting angle if not give.
        '''
        if counting_angle is not None:
            theta = math.cos(math.radians(counting_angle))
        else:
            n = self.dataset_to_contour.n_entries
            theta = (n - 1.0) / (n + 1.0)

        datac = self.dataset_to_contour.data
        self.color_data = np.where(np.abs(np.dot(self.nodes, datac.T))
                                   >= theta, 1, 0).sum(axis=1) / n * 100
        return self.color_data

    def _count_fisher(self, k=None):
        '''
        Perform data counting as in Robin and Jowett (1986).
        Will guess an appropriate k if not given.
        '''
        if k is None:
            k = self._optimize_k
