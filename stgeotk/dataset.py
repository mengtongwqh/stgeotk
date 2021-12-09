from abc import ABC, abstractmethod
import numpy as np

from . utility import log_info
from . stereomath import cartesian_to_line, line_to_cartesian


class DatasetBase(ABC):
    def __init__(self, spacedim, **kwargs):
        self._spacedim = spacedim
        self._data = None
        self._data_legend = ""
        self._color_data = None
        self._color_legend = ""
        self._n_entries = 0
        # whether the vector direction matters
        self._is_polarized = kwargs.get("polarized", False)
        if self._is_polarized and "hemisphere" in kwargs:
            raise RuntimeError("Polarized data cannot be"
                               "constrained to a hemisphere")
        self._hemisphere = kwargs.get("hemisphere", "lower")

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
        if value is None:
            return
        if len(value) != self.n_entries:
            raise RuntimeError("The dataset has length {0} "
                               "but the color data has length {1}".
                               format(self.n_entries, len(value)))
        self._color_data = value

    @abstractmethod
    def _set_data(self, value):
        pass

    def _do_data_load(self, data, data_legend, color_data, color_legend):
        '''
        load the dataaset
        '''
        self.data = data
        self._data_legend = data_legend
        self.color_data = color_data
        self._color_legend = color_legend


class LineData(DatasetBase):
    '''
    Stores linear structural data (or lineation),
    in the form of directional cosines.
    '''

    def __init__(self, **kwargs):
        super().__init__(3, **kwargs)

    def load_data(self,
                  data, data_legend="Lineation",
                  color_data=None, color_legend=""):
        '''
        Load the dataset
        '''
        self._do_data_load(data, data_legend, color_data, color_legend)
        log_info("LineData of {0} entries are loaded with legend \"{1}\"".
                 format(self.n_entries, self.data_legend))

    def _set_data(self, value):
        '''
        Parse the dataset entries.
        The entries can either be in the vector form of n_entries x 3
        or in the trend/plunge form of n_entries x 2 (in degree)
        '''
        if isinstance(value, list):
            # if value is a builtin list, convert to numpy array
            value = np.array(value, dtype=np.double)
        if value.shape[1] == 2:  # trend/plunge format
            self._data = line_to_cartesian(value)
        elif value.shape[1] == 3:
            # normalize to unit vector
            self._data = value / np.linalg.norm(value, axis=1)[:, np.newaxis]
            if not self.is_polarized:
                self.__restrict_to_hemisphere(self._hemisphere)
        else:
            raise(RuntimeError(
                "Unexpected dimension for input dataset: {0}".format(value.shape)))
        self._n_entries = self.data.shape[0]

    def __restrict_to_hemisphere(self, hemisphere):
        '''
        Restrict non-directional dataset to the upper/lower hemisphere
        '''
        if __debug__ and self._spacedim != 3:
            raise RuntimeError(
                "Hemisphere restriction only makes sense for 3d dataset.")
        if hemisphere == "lower":
            upper_hemis = (self._data[:, 2] > 0)
            self._data[upper_hemis, :] = -self._data[upper_hemis, :]
        elif hemisphere == "upper":
            lower_hemis = self._data[:, 2] < 0
            self._data[lower_hemis, :] = -self._data[lower_hemis, :]
        else:
            raise RuntimeError(
                "Unknown hemisphere type {0}".format(hemisphere))

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
                f"Export method for {type(self._data).__name__} is not implemented")


class PlaneData(DatasetBase):
    '''
    Store the planar structural data (or foliation/bedding),
    in the form of directional cosines of the normal
    '''

    def __init__(self, **kwargs):
        super().__init__(3, **kwargs)
        if self.is_polarized or self._hemisphere != "lower":
            raise RuntimeError("Plane data must be unpolarized and lower-hemispheric")


    def load_data(self,
                  data, data_legend="Foliation",
                  color_data=None, color_legend=""):
        self._do_data_load(data, data_legend, color_data, color_legend)
        log_info("PlaneData of {0} entries are loaded with legend \"{1}\"".
                 format(self.n_entries, self.data_legend))

    def _set_data(self, value):
        # the underlying data is stored in strike-dip format
        if isinstance(value, list):
            # builtin list, convert to numpy array
            value = np.array(value, dtype = np.double)
        elif isinstance(value, np.ndarray):
            if value.shape[1] == 2:
                # assuming this is already in strike-dip format
                self._data = value
            elif value.shape[1] == 3:
                # the pole of the plane
                # Note that this is assuming lower hemisphere
                self._data = cartesian_to_line(value)
        else:
            raise RuntimeError(f"Input type {type(value).__name__} is unexpected")
        self._n_entries  = self.data.shape[0]


    def write_to_file(self, **kwargs):
        '''
        write the underlying data to file
        '''
        file = kwargs.get("file", self.data_legend + ".txt")
        if isinstance(self.data, np.ndarray):
            np.savetxt(file, self.data)
            log_info(
                f"{self._n_entries} planar data entries written to file [{file}]")
        else:
            raise RuntimeError("Export method for "\
            f"{type(self._data).__name__} is not implemented")
