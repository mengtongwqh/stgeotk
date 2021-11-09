from . utility import *
from . coordinates import *

import numpy as np
import math
from abc import ABC, abstractmethod


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
        # ordinary points on sphere
        for phi in np.arange(degree_spacing, 90.0, degree_spacing):
            azm_spacing = math.sin(spacing/2.0) / math.sin(math.radians(phi))
            azm_spacing = math.degrees(2*math.asin(azm_spacing))
            for theta in np.arange(0.0, 360.0, azm_spacing):
                nodes.append((theta + phi + azm_spacing/2.0, 90.0 - phi))
        # equator
        for theta in np.arange(0., 360.,  azm_spacing):
            nodes.append((theta + 90.0 + degree_spacing / 2.0, 0.0))
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
        self._is_polarized = False  # whether the vector direction matters

    @property
    def spacedim(self):
        return self._spacedim

    @property
    def n_entries(self):
        return self._n_entries

    @property
    def is_polarized(self):
        return self._is_polarized

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
        self._is_polarized = False
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
        log_info("LineData of {0} entries are loaded with legend \"{1}\"".
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

    def write_to_file(self, **kwargs):
        '''
        write the underlying data to file so that comparisons
        can be made with other stereonet software
        '''
        file = kwargs.get("file", self.data_legend + ".txt")
        if isinstance(self.data, np.ndarray):
            # convert to trend-plunge (geological) coordinates
            line_tp = cartesian_to_line(self._data)
            np.savetxt(file, line_tp)
            log_info(
                f"{self._n_entries} line data entries written to file [{file}]")
        else:
            raise RuntimeError(
                f"Export method for {type(self._data).name()} is not implemented")


class ContourData(DatasetBase):
    '''
    Object for computing the density contour.
    '''

    def __init__(self, data_to_contour=None, **kwargs):
        # counting grid
        super().__init__(3)
        # parse graphing options
        self.parse_options(kwargs)
        # load the data
        self._color_data = None
        if data_to_contour is not None:
            self.load_data(data_to_contour)

    def parse_options(self, opts):
        self._grid_spacing = opts.get("grid_spacing", 2.5)
        # counting options
        self._counting_method = opts.get("counting_method", "fisher")
        self._counting_angle = opts.get("counting_angle", None)
        self._counting_k = opts.get("counting_k", None)
        self._auto_k_optimization = opts.get("auto_k_optimization", False)
        self._n_sigma = opts.get("n_sigma", 3)

    def load_data(self, data_to_contour):
        # load the actual data
        self._is_polarized = data_to_contour.is_polarized

        if isinstance(data_to_contour, LineData):
            if data_to_contour.n_entries == 0:
                raise RuntimeError(
                    "The input LineData is empty and therefore "
                    "cannot be contoured.")
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

        log_info(
            f"{type(data_to_contour).__name__} {data_to_contour.data_legend}"
            f"loaded for contouring. "
            f"Polarized = {data_to_contour.is_polarized}")

    @property
    def dataset_to_contour(self):
        return self._dataset_to_contour

    @property
    def counting_method(self):
        return self._counting_method

    @property
    def nodes(self):
        return self._data

    def count(self):
        if not np.all(np.isfinite(self.dataset_to_contour.data)):
            raise RuntimeError("Data to be contoured has infinite values.")

        timer = Timer()
        timer.start()
        if self._counting_method == "kamb":
            count = self._count_kamb(self._counting_angle)
        elif self._counting_method == "fisher":
            count = self._count_fisher(self._counting_k)
        else:
            raise RuntimeError(
                "Unknown counting method {0}".format(self._counting_method))
        timer.stop()
        log_info("ContourData density range [{0}, {1}] using method \"{2}\".".format(
            min(count), max(count), self.counting_method))
        return count

    def _set_data(self, value):
        raise RuntimeError(
            "Data attributes in ContourData is set automatically. "
            "Do not use this setter.")

    def _optimize_k(self):
        '''
        Optimizes the value of K from the data, 
        Inherited from auttitude package,
        where they claimed this is from using Diggle and Fisher (88).
        However that publication does not exist and 
        we don't know why this works.
        '''
        timer = Timer()
        timer.start()
        from scipy.optimize import minimize_scalar
        datac = self.dataset_to_contour.data

        # objective function to be minimized
        # TODO consider rewriting this using NewtonRaphson
        def obj(k):
            # suppress warning if k == 0.0
            k = 1.0e-9 if abs(k) < 1.0e-9 else k
            W = np.exp(k * (np.abs(np.dot(datac, datac.T)))) \
                * (k / (4.0 * math.pi * math.sinh(k)))
            np.fill_diagonal(W, 0.0)
            return -np.log(W.sum(axis=0)).sum()

        val = minimize_scalar(obj).x
        timer.stop()
        return val

    def _count_kamb(self, counting_angle):
        '''
        Performs counting as in Robin and Jowett (1986), 
        which is in turn based on Kamb (1956). 
        Will estimate an appropriate counting angle if not give.
        Estimations comes from Robin and Jowett (1986), Table 2.
        https://doi.org/10.1016/0040-1951(86)90044-2
        '''
        datac = self.dataset_to_contour.data
        n = self.dataset_to_contour.n_entries
        polarized = self.dataset_to_contour.is_polarized

        if counting_angle is not None:
            cos_theta = math.cos(math.radians(counting_angle))
            log_info("Prescribed counting angle for the Kamb's method is {0} degrees".
                     format(counting_angle))
        else:
            nsig2 = self._n_sigma * self._n_sigma
            if polarized:
                cos_theta = (n - nsig2) / (n + nsig2)
            else:
                cos_theta = n / (n + nsig2)
            log_info("Counting angle for the Kamb's method is chosen to be {0} degrees for {1} standard deviations".
                     format(math.degrees(math.acos(cos_theta)), self._n_sigma))

        # counting loop
        self.color_data = np.where(np.abs(np.dot(self.nodes, datac.T))
                                   >= cos_theta, 1, 0).sum(axis=1) / n * 100.0
        return self.color_data

    def _count_fisher(self, k=None):
        '''
        Perform data counting as in Robin and Jowett (1986) 
        with spherical distribution
        Will guess an appropriate k if not given.
        '''
        timer = Timer()
        timer.start()
        n = self.dataset_to_contour.n_entries
        polarized = self.dataset_to_contour.is_polarized

        if k is None:
            if self._auto_k_optimization:
                k = self._optimize_k()
                log_info(
                    "Sample-optimized Fisher counting k = {0}".format(k))
            else:
                nsig2 = self._n_sigma * self._n_sigma
                if polarized:
                    k = 1.0 + n/nsig2
                else:
                    k = 2.0 * (1.0 + n/nsig2)
                log_info(
                    "Fisher counting k = {0} selected for {1} standard deviations".format(k, self._n_sigma))
        else:
            log_info("Prescribed Fisher counting k = {0}".format(k))

        datac = self.dataset_to_contour.data
        self.color_data = np.exp(
            k * (np.abs(np.dot(self.nodes, datac.T)) - 1.0)).sum(axis=1) / n * 100.0

        timer.stop()
        return self.color_data
