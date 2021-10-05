import numpy as np
from . utility import line_to_cartesian
from abc import ABC, abstractmethod
import warnings


class DatasetBase(ABC):
    def __init__(self, spacedim):
        self._spacedim = spacedim
        self._data = []
        self._data_legend = ""
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

    @data.setter
    def data(self, value):
        self._set_data(value)

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
        # default options
        self._lower_hemisphere = True
        self._color_data = []
        self._color_legend = ""
        if "lower_hemisphere" in kwargs:
            self._lower_hemisphere = kwargs["lower_hemisphere"]

    @property
    def color_data(self):
        if not self._color_data:
            warnings.warn("Data is empty", RuntimeWarning)
        return self._color_data

    @color_data.setter
    def color_data(self, value):
        if value is None:
            return
        if len(value) != self._n_entries:
            raise RuntimeError(
                "Color data must have same number of entries as data")
        self._color_data = value

    def load_data(self,
                  data, data_legend="Lineation",
                  color_data=None, color_legend=""):
        '''
        Load the dataset
        '''
        self.data = data
        self._data_legend = data_legend
        self.color_data = color_data
        self._color_legend = color_legend

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
                "Unexpected dimension for input dataset: " + value.shape))
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
