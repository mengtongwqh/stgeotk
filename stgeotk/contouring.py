import concurrent.futures
import math
import os
import multiprocessing
import numpy as np
from scipy.optimize import minimize_scalar

from . stereomath import line_to_cartesian
from . dataset import DatasetBase, LineData
from . utility import Timer, log_info


def _contour_data_variance_chunk(datac, k, alpha, idx_begin, idx_end):
    # unroll_interval = 1000
    # idx = np.append(np.arange(idx_begin, idx_end, unroll_interval), idx_end)
    # for i in range(0, len(idx)-1):
    #     i_begin, i_end = idx[i], idx[i+1]
    #     tmp = np.exp(
    #         k * np.abs(np.dot(datac, datac[i_begin:i_end, :].T)) + alpha)
    #     np.fill_diagonal(tmp, 0.0)
    #     s += -np.log(tmp.sum(axis=0)).sum()
    s = 0.0
    for i in range(idx_begin, idx_end):
        tmp = np.exp(k * np.abs(np.dot(datac, datac[i, :])) + alpha)
        tmp[i] = 0.0
        s += -np.log(tmp.sum())
    return s


def _contour_data_variance(datac, k):
    # log_info(f"[{current_function_name()}]: k = {k}")
    k = 1.0e-9 if abs(k) < 1.0e-9 else k
    if k < 500.0:
        alpha = math.log(k/4.0/math.pi/math.sinh(k))
    else:
        alpha = math.log(k/2.0/math.pi) - k

    n_procs = multiprocessing.cpu_count()
    idx = (datac.shape[0] + np.arange(0, n_procs))//n_procs
    idx = np.concatenate(([0], np.cumsum(idx)))
    s = 0.0

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(_contour_data_variance_chunk, \
                                   datac, k, alpha, idx[iproc], idx[iproc+1]) \
                   for iproc in range(0, n_procs)]
        for chunk in concurrent.futures.as_completed(results):
            s += chunk.result()
    return s


class ContourData(DatasetBase):
    """
    Object for computing the density contour.
    """

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
            f" loaded for contouring. "
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
        """
        Optimizes the value of K from the data,
        Inherited from auttitude package,
        where they claimed this is from using Diggle and Fisher (88).
        However that publication does not exist and
        we don't know why this works.
        """
        use_faster_version = False
        # Windows does not fork, will cause problems in QGIS
        use_multiprocessing = (os.name == "posix")
        datac = self.dataset_to_contour.data

        with Timer() as _:
            if use_faster_version:
                M = np.abs(np.dot(datac, datac.T))
                np.fill_diagonal(M, 0.0)

                def obj(k):
                    with Timer() as _:
                        k = 1.0e-9 if abs(k) < 1.0e-9 else k
                        # log_info("k = {0}".format(k))
                        if k < 500.0:
                            W = np.exp(M*k + math.log(k /
                                                      4.0/math.pi/math.sinh(k)))
                        else:
                            W = np.exp(M*k + math.log(k/4.0/math.pi) - k)
                        np.fill_diagonal(W, 0.0)
                        val = -(np.log(W.sum(axis=0))).sum()
                    return val

            elif use_multiprocessing:
                def obj(k):
                    return _contour_data_variance(datac, k)
            else:
                def obj(k):
                    with Timer() as _:
                        # suppress warning if k == 0.0
                        k = 1.0e-9 if abs(k) < 1.0e-9 else k
                        W = np.exp(k * (np.abs(np.dot(datac, datac.T)))) \
                            * (k / (4.0 * math.pi * math.sinh(k)))
                        np.fill_diagonal(W, 0.0)
                        val = -np.log(W.sum(axis=0)).sum()
                    return val
            val = minimize_scalar(obj).x
        return val

    def _count_kamb(self, counting_angle):
        """
        Performs counting as in Robin and Jowett (1986),
        which is in turn based on Kamb (1956).
        Will estimate an appropriate counting angle if not give.
        Estimations comes from Robin and Jowett (1986), Table 2.
        https://doi.org/10.1016/0040-1951(86)90044-2
        """
        datac = self.dataset_to_contour.data
        n = self.dataset_to_contour.n_entries
        polarized = self.dataset_to_contour.is_polarized

        if counting_angle is not None:
            cos_theta = math.cos(math.radians(counting_angle))
            log_info("Prescribed counting angle for the Kamb's method is"\
                     "{0} degrees".format(counting_angle))
        else:
            nsig2 = self._n_sigma * self._n_sigma
            if polarized:
                cos_theta = (n - nsig2) / (n + nsig2)
            else:
                cos_theta = n / (n + nsig2)
            log_info("Counting angle for the Kamb's method is chosen to be {0}"
                     "degrees for {1} standard deviations".
                     format(math.degrees(math.acos(cos_theta)), self._n_sigma))

        # counting loop
        self.color_data = np.where(np.abs(np.dot(self.nodes, datac.T))
                                   >= cos_theta, 1, 0).sum(axis=1) / n * 100.0
        return self.color_data

    def _count_fisher(self, k=None):
        """
        Perform data counting as in Robin and Jowett (1986)
        with spherical distribution
        Will guess an appropriate k if not given.
        """
        timer = Timer()
        timer.start()
        n = self.dataset_to_contour.n_entries
        polarized = self.dataset_to_contour.is_polarized

        if k is None:
            if self._auto_k_optimization:
                log_info("Requested auto optimization of Fisher-k ...")
                k = self._optimize_k()
                log_info(f"Sample-optimized Fisher counting k = {k}")
            else:
                nsig2 = self._n_sigma * self._n_sigma
                if polarized:
                    k = 1.0 + n/nsig2
                else:
                    k = 2.0 * (1.0 + n/nsig2)
                log_info(
                    "Fisher counting k = {0} selected for {1}"
                    "standard deviations".format(k, self._n_sigma))
        else:
            log_info("Prescribed Fisher counting k = {0}".format(k))

        datac = self.dataset_to_contour.data
        self.color_data = np.exp(
            k * (np.abs(np.dot(self.nodes, datac.T)) - 1.0)).sum(axis=1) / n * 100.0

        timer.stop()
        return self.color_data


class CountingGrid:
    """
    Generate a counting grid for density calculation.
    """

    def __init__(self, grid_type):
        self._grid_type = grid_type

    def _hemispherical_grid(self, degree_spacing):
        """
        Construct the counting grid in the lineation coordinate
        Assuming lower hemispheric
        """
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
        """
        Construct the counting grid for the whole sphere
        """
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

    def generate(self, degree_spacing=2.5):
        if self._grid_type == "hemispherical":
            return self._hemispherical_grid(degree_spacing)
        elif self._grid_type == "spherical":
            return self._spherical_grid(degree_spacing)
        else:
            raise RuntimeError(f"Unknown counting grid type {self._grid_type}")
