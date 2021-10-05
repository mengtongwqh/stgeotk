from math import sqrt
import numpy as np
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from . dataset import DatasetBase
from abc import ABC, abstractmethod


class ProjectionBase(ABC):
    def project(self, data):
        if isinstance(data, DatasetBase):
            x, y, z = data.data.T
        elif isinstance(data, np.ndarray):
            x, y, z = data.T
        else:
            raise RuntimeError(f"Unexpected data type {type(data)}")
        return np.array(self._do_project(x, y, z)).T

    def inverse(self, data):
        raise NotImplementedError()

    @abstractmethod
    def _do_project(self, x, y, z):
        return


class EqualArea(ProjectionBase):
    def _do_project(self, x, y, z):
        # normalized so that the lower hemisphere
        # projects to unit circle
        alpha = np.sqrt(1.0 / (1.0 - z))
        X = x * alpha
        Y = y * alpha
        return x * alpha, y * alpha


class EqualAngle(ProjectionBase):
    def _do_project(self, x, y, z):
        alpha = 1.0 / (1.0 - z)
        return x * alpha, y * alpha


class Stereonet:
    '''
    Base class for all plotting on stereonet.
    '''

    def __init__(self, fig=None, ax=None, **kwargs):
        self.plots = []
        if fig is None and ax is None:
            self.figure = plt.figure(figsize=(8, 5), facecolor="white")
            self.data_axes = self.figure.add_axes([0.01, 0.01, 0.6, 0.98],
                                                  xlim=(-1.1, 1.2),
                                                  ylim=(-1.1, 1.1),
                                                  adjustable="box",
                                                  clip_on="True",
                                                  autoscale_on="False")
            # self.color_axes = self.figure.add_axes([0.7, 0.1, 0.02, 0.4])
            self.data_axes.set_aspect(
                aspect="equal", adjustable=None, anchor="W")
        elif fig is not None and ax is not None:
            self.figure, self.data_axes = fig, ax
        else:
            raise RuntimeError("Figure and axis must be provided in pairs")
        self.clear()
        self.datasets = []
        self._projection_method = "equal_area"
        if "projection_method" in kwargs:
            self._projection_method = kwargs["equal_area"]
        else:
            self.projection = "equal_area"

    @property
    def projection(self):
        return self._projection

    @projection.setter
    def projection(self, value):
        if isinstance(value, str):
            if (value == "equal_area"):
                self._projection = EqualArea()
            elif (value == "equal_angle"):
                self._projection = EqualAngle()
            else:
                raise RuntimeError("Unknown projection method.")
        elif isinstance(value, ProjectionBase):
            self._projection = value
        else:
            raise RuntimeError(f"Unknown projection input {type(value)}")

    def clear(self):
        self.plots = []
        self.data_axes.set_axis_off()
        self.data_axes.set_aspect(
            aspect="equal", adjustable=None, anchor="SW")
        self.draw_primitive_circle()

    def append_plot(self, plot, **kwargs):
        '''
        Append one set of data into the current plot
        '''
        self.plots.append(plot)

    def generate_plots(self):
        '''
        Plot each and every dataset in this stereonet plot
        '''
        for plot in self.plots:
            plot.draw()

    def draw_primitive_circle(self):
        '''
        Draw the primitive circle 
        '''
        self.data_axes.set_axis_off()
        circ = Circle((0, 0), radius=1, edgecolor="black",
                      facecolor="none", clip_box="None")
        self.data_axes.add_patch(circ)
        return circ

    def project(self, data):
        '''
        Project the xyz data to a 2d plane
        using the suitable projection method
        '''
        return self.projection.project(data)


class LinePlot:
    '''
    Plot the lineation data on the stereonet
    '''

    def __init__(self, stereonet, data, **kwargs):
        self.stereonet = stereonet
        self.data = data
        self.graphing_options = kwargs

    def draw(self):
        '''
        Execute plotting on the stereonet.
        '''
        x, y = self.stereonet.project(self.data).T
        return self.stereonet.data_axes.scatter(x, y, **self.graphing_options)


class ContourPlot:
    def __init__(self, stereonet, contour_data):
        self.stereonet = stereonet
        self.data = contour_data

    def draw(self):
        X, Y = self.stereonet.project(self.data.nodes).T
        count = self.data.count_kamb() / self.data.data.n_entries
        interval = np.linspace(0, count.max(), 20)
        return self.stereonet.data_axes.tricontourf(X, Y, count, interval)


# class LineationPlot(StereoPlot):
#     '''
#     Plotting linear data as points on the stereonet.
#     '''

#     def __init__(self, fig=None, ax=None, **kwargs):
#         self.title = None
#         self.equal_area = True

#         if fig is None and ax is None:
#             self.fig = plt.figure()
#             self.ax = self.fig.add_subplot(111, projection='polar')
#         elif fig is not None and ax is not None:
#             self.fig, self.ax = fig, ax
#         else:
#             raise RuntimeError("figure and axis must be provided in pairs")
#         self.set_properties(kwargs)

#     def set_properties(self, **kwargs):
#         for key, value in kwargs.items():
#             if key == "marker":
#                 self.marker = value
#             elif key == "title":
#                 self.title = value
#             elif key == "equal_area":
#                 if value:
#                     self.equal_area = True
#                 else:
#                     self.equal_area = False
#             else:
#                 raise RuntimeError(
#                     "Unknown properties in class LineationPlot: " + key)

#     def load_data(self, x, y, z, color=None):
#         '''
#         load the data into the LineationPlot object
#         '''
#         sphere = cartesian_to_spherical(x, y, z)
#         self.theta = sphere[0]
#         if self.equal_area:
#             self.r = equal_area_projection(sphere[1])
#         else:
#             self.r = equal_angle_projection(sphere[1])
#         self.color = color

#     def plot(self, **kwargs):
#         '''
#         \param[in] x,y,z 3d components of the vector
#         '''
#         cmap = 'coolwarm_r'
#         marker = '+'

#         for key, value in kwargs.items():
#             if key == "cmap":
#                 cmap = value
#             if key == "marker":
#                 marker = value

#         if self.color is not None and self.color.size != 0:
#             stereo_plot = self.ax.scatter(self.theta, self.r, marker=marker,
#                                           c=self.color, cmap=cmap)
#             self.fig.colorbar(stereo_plot)
#         else:
#             stereo_plot = self.ax.scatter(self.theta, self.r, marker=marker)

#         self.ax.set_title(self.title)
#         self.ax.text(0.95, 0.05, f"N = {self.theta.size}",
#                      horizontalalignment='center', transform=self.ax.transAxes)
#         self.__set_stereonet_format()

#         return stereo_plot

#     def draw_contour(self, n_bins_theta=72, n_bins_r=16):
#         cmap = "coolwarm"
#         # r_eps = 1.0 / n_bins_r / 1e4
#         r_bins = equal_area_projection(
#             np.linspace(0.5 * np.pi, 0,  n_bins_r + 1))
#         # r_bins = np.append(r_bins, 1.0)
#         # r_bins = np.insert(r_bins, 0, 0.0, axis=0)

#         d_theta = 2 * np.pi / n_bins_theta
#         theta_bins = np.linspace(0.0,  2*np.pi + d_theta, n_bins_theta + 1)

#         H, theta_edges, r_edges = np.histogram2d(
#             self.theta, self.r, bins=(theta_bins, r_bins))
#         print(np.sum(H))
#         H[-1, :] = H[0, :]

#         # plot data in the middle of the bins
#         r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
#         # r_mid[0] = 0.0
#         # r_mid[-1] = 1.0
#         theta_mid = 0.5 * (theta_edges[:-1] + theta_edges[1:])
#         print(theta_mid)
#         cax = self.ax.contour(theta_mid, r_mid, H.T, 10, cmap=cmap)
#         self.fig.colorbar(cax)

#     def __set_stereonet_format(self):
#         self.ax.set_theta_zero_location("N")
#         self.ax.set_theta_direction(-1)
#         self.ax.set_rmin(0.0)
#         self.ax.set_rmax(1.0)
#         self.ax.tick_params(direction="inout")
#         self.ax.set_xticks([0.0])
#         self.ax.set_xticklabels(["N"])
#         self.ax.set_yticklabels([])
#         self.ax.grid(False)
if __name__ == "__main__":
    x = np.array([1.0, 2.0, 1.0])
    y = np.array([0.5, 0.3, 0.6])
    z = np.array([-1.02, 1.0, 0.5])

    plot_graph = Stereonet()
    plt.show()
    # plot_graph.draw_primitive_circle()

    # azm, plg, rho = cartesian_to_spherical(x, y, z)
    # print(f"{azm},{plg},{rho}")

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="polar")

    # r = equal_area_projection(plg)

    # ax.scatter(azm, r)
    # ax.set_theta_zero_location("N")
    # ax.set_theta_direction(-1)
    # ax.set_rmin(0.0)
    # ax.set_rmax(1.0)
    # ax.tick_params(direction="inout")
    # ax.set_xticks([0.0])
    # ax.set_xticklabels(["N"])
    # ax.set_yticklabels([])
    # ax.grid(False)
    # plt.show()
