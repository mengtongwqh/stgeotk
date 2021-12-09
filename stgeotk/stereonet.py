from abc import ABC, abstractmethod
import numpy as np
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from . dataset import DatasetBase
from . contouring import ContourData
from . utility import Timer, log_info
from . stereomath import plane_nodes

# fonts
title_font = {"family": "sans-serif", "size": "12", "weight": "regular",
              "color": "black", "verticalalignment": "bottom"}
label_font = {"family": "sans-serif", "size": "10",
              "color": "black", "verticalalignment": "bottom"}
info_font = {"family": "sans-serif", "size": "6",
             "color": "black", "verticalalignment": "top", "horizontalalignment": "left"}


class ProjectionBase(ABC):
    """
    Abstract for all projections
    """

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
    """
    Lambert equal area projection.
    The lower-hemisphere is normalized to unit radius
    """

    def _do_project(self, x, y, z):
        # normalized so that the lower hemisphere
        # projects to unit circle
        alpha = np.sqrt(1.0 / (1.0 - z))
        return x * alpha, y * alpha


class EqualAngle(ProjectionBase):
    """
    Schmidt equal angle projection.
    """

    def _do_project(self, x, y, z):
        alpha = 1.0 / (1.0 - z)
        return x * alpha, y * alpha


class Stereonet:
    """
    The stereonet object will accept different plot objects
    and plot them on a single stereonet,
    with the presribed projection method.
    """

    def __init__(self, fig=None, ax=None, **kwargs):
        self.plots = []
        self.color_axes = {}  # {id(plot) : color_axis } pair

        self._caxes_origin = [0.6, 0.05]
        self._caxes_current_origin = self._caxes_origin
        self._caxes_extent = [0.02, 0.4]

        if fig is None and ax is None:
            self.figure = plt.figure(figsize=(10, 6), facecolor="white")
            self.data_axes = self.figure.add_axes([0.0, 0.0, 0.6, 0.98],
                                                  xlim=(-1.1, 1.2),
                                                  ylim=(-1.1, 1.1),
                                                  adjustable="box",
                                                  clip_on="True",
                                                  autoscale_on="False")
            self.clear()
        elif fig is not None and ax is not None:
            self.figure, self.data_axes = fig, ax
        else:
            raise RuntimeError("Figure and axis must be provided in pairs")
        if "projection_method" not in kwargs:
            self.projection = "equal_area"

    @property
    def projection(self):
        return self._projection

    @projection.setter
    def projection(self, value):
        if isinstance(value, str):
            if value == "equal_area":
                self._projection = EqualArea()
            elif value == "equal_angle":
                self._projection = EqualAngle()
            else:
                raise RuntimeError(
                    "Unknown projection method {0}.".format(value))
        elif isinstance(value, ProjectionBase):
            self._projection = value
        else:
            raise RuntimeError(
                "Unknown projection input {0}".format(type(value)))

    def clear(self):
        """
        Remove all plots that are connected to the stereonet
        and redraw the primitive circle
        """
        self.plots = []
        self.data_axes.set_axis_off()
        self.data_axes.set_aspect(
            aspect="equal", adjustable=None, anchor="W")
        self.draw_primitive_circle()

    def append_plot(self, plot):
        """
        Append one data plot into the current plot.
        If there is color info in the plot, request a color axis.
        """
        self.plots.append(plot)

        # register this data
        if id(plot) in self.color_axes:
            raise RuntimeError(
                "Plot with legend {0} has already been added.".
                format(plot.data_legend))

        if plot.dataset_to_plot.color_data is None:
            # color axis is None
            self.color_axes[id(plot)] = None
        else:
            self.color_axes[id(plot)] = self._create_next_color_axis()

        log_info("{0} added to stereonet with options:{1}".format(
            type(plot).__name__, plot.plot_options))

    def generate_plots(self, show_plot=True):
        """
        Plot all datasets on this stereonet
        """
        timer = Timer()
        timer.start()
        for plot in self.plots:
            plot_obj = plot.draw()
            caxis = self.color_axes[id(plot)]
            if caxis is not None:
                cb = plt.colorbar(plot_obj, cax=caxis,
                                  orientation="vertical", spacing="proportional")
                cb.formatter.set_powerlimits((0, 0))
                cb.set_label(plot.dataset_to_plot.color_legend)
                caxis.yaxis.set_label_position("left")

        # draw legends
        self.data_axes.legend(loc="upper right")
        timer.stop()
        log_info("All plots are successfully generated")
        # info string
        info_txt = ""
        for plot in self.plots:
            info_txt += plot.info_text() + '\n'
        self.data_axes.text(1.05, 0.95, info_txt,
                            transform=self.data_axes.transAxes,  **info_font)
        # show plots immediately
        if show_plot:
            plt.show()

    def _create_next_color_axis(self):
        position = self._caxes_current_origin.copy()
        position.extend(self._caxes_extent)
        caxis = self.figure.add_axes(position, anchor="SW")
        self._caxes_current_origin[0] += self._caxes_extent[0] + 0.1
        return caxis

    def draw_primitive_circle(self):
        """
        Draw the primitive circle
        """
        self.data_axes.set_axis_off()
        circ = Circle((0, 0), radius=1, edgecolor="black",
                      facecolor="none", clip_box="None")
        self.data_axes.add_patch(circ)
        self.data_axes.text(0.0, 1.02, "N", **title_font,
                            horizontalalignment="center")

        # crosses for each quadrant and center
        x_cross = [0, 1, 0, -1, 0]
        y_cross = [0, 0, 1, 0, -1]
        self.data_axes.plot(x_cross, y_cross, "k+", markersize=8)
        return circ

    def project(self, data):
        """
        Project the xyz data to a 2d plane
        using the suitable projection method
        """
        return self._projection.project(data)

    def save_plot(self, fname, **kwargs):
        """
        export the plot to disk
        """
        self.figure.savefig(fname, **kwargs)
        log_info(f"Stereonet plot is saved to {fname}")


class PlotBase(ABC):
    """
    Base class for all plot objects
    """

    def __init__(self, stereonet, data, **kwargs):
        self._stereonet = stereonet
        self._dataset_to_plot = data
        self._plot_options = dict()
        self.plot_options = kwargs

    @ abstractmethod
    def _set_plot_options(self, options):
        return

    @ abstractmethod
    def draw(self):
        """
        This method will be called by stereonet
        to generate the plots
        """
        return

    @ abstractmethod
    def info_text(self):
        return

    @ property
    def stereonet(self):
        return self._stereonet

    @ property
    def dataset_to_plot(self):
        return self._dataset_to_plot

    @ property
    def plot_options(self):
        return self._plot_options

    @ plot_options.setter
    def plot_options(self, value):
        self._set_plot_options(value)

    @staticmethod
    def extract_option(opt_copy, key):
        if key in opt_copy:
            return opt_copy.pop(key)
        raise RuntimeError(
            "{0} is not in options {1}".format(key, opt_copy))


class LinePlot(PlotBase):
    """
    Plot the lineation data on the stereonet
    """

    def _set_plot_options(self, options):
        self._plot_options = options
        # set default options
        if "linewidths" not in self._plot_options:
            self._plot_options["linewidths"] = 1.0
        if "marker" not in self._plot_options:
            self._plot_options["marker"] = '+'
        if "cmap" not in self._plot_options:
            self._plot_options["cmap"] = "coolwarm"

    def draw(self):
        """
        Execute plotting on the stereonet.
        """
        opt = self.plot_options.copy()
        x, y = self.stereonet.project(self.dataset_to_plot).T
        # extract color data, if we have them:
        if self.dataset_to_plot.color_data is not None:
            if "color" in opt:
                opt.pop("color")
            plot = self.stereonet.data_axes.scatter(
                x, y, c=self.dataset_to_plot.color_data,
                label=self.dataset_to_plot.data_legend,
                **opt)
        else:
            if "color" not in opt:
                opt["color"] = "black"
            plot = self.stereonet.data_axes.scatter(x, y,
                                                    label=self.dataset_to_plot.data_legend,  **opt)
        return plot

    def info_text(self):
        return "Scatter plot for dataset \"{0}\" contains {1} points".format(
            self.dataset_to_plot.data_legend, self.dataset_to_plot.n_entries)


class PlanePlot(PlotBase):
    """
    Plot the planar data as great circles on the stereonet
    """

    def _set_plot_options(self, options):
        self._plot_options = options
        # set default options
        if "n_segments" not in self._plot_options:
            self._plot_options["n_segments"] = 90  # plane resolution
        if "color" not in self._plot_options:
            self._plot_options["color"] = "black"
        if "linewidth" not in self._plot_options:
            self._plot_options["linewidth"] = 0.5

    def draw(self):
        """
        draw the big circles onto the stereonet,
        entry by entry.
        """
        dataset = self.dataset_to_plot.data
        opt = self.plot_options.copy()
        n_segments = opt.pop("n_segments")
        ax = self.stereonet.data_axes
        data_label = self.dataset_to_plot.data_legend

        first_flag = True
        for data in dataset:
            nodes = plane_nodes(data[0], data[1], n_segments)
            x, y = self.stereonet.project(nodes).T
            if first_flag:
                ax.plot(x, y, label=data_label, **opt)
                first_flag = False
            else:
                ax.plot(x, y, **opt)

    def info_text(self):
        info = "Big circle plot for dataset \"{0}\" contains {1} entries.".\
            format(self.dataset_to_plot.data_legend,
                   self.dataset_to_plot.n_entries)
        return info


class ContourPlot(PlotBase):
    """
    Plotting the contour data
    """

    def __init__(self, stereonet, contour_data, **kwargs):
        super().__init__(stereonet, contour_data, **kwargs)
        if not isinstance(contour_data, ContourData):
            raise RuntimeError("Only ContourData is allowed for ContourPlot.")

    def draw(self):
        opt = dict(self.plot_options).copy()

        X, Y = self.stereonet.project(self.dataset_to_plot.nodes).T
        count = self.dataset_to_plot.color_data

        # make a copy of the options
        interval = np.linspace(
            0, count.max(),
            self.extract_option(opt, "n_intervals"))

        if self.extract_option(opt, "filled"):
            plot = self.stereonet.data_axes.tricontourf(
                X, Y, count, interval, **opt)
        else:
            plot = self.stereonet.data_axes.tricontour(
                X, Y, count, interval, **opt)
        return plot

    def _set_plot_options(self, options):
        self._plot_options = options
        # set default options
        if "n_intervals" not in self._plot_options:
            self._plot_options["n_intervals"] = 10
        if "filled" not in self._plot_options:
            self._plot_options["filled"] = True
        if "cmap" not in self._plot_options:
            self._plot_options["cmap"] = "Oranges"
        if "antialiased" not in self._plot_options:
            self._plot_options["antialiased"] = False
        if "alpha" not in self._plot_options:
            self._plot_options["alpha"] = 1.0

    def info_text(self):
        return "ContourPlot of dataset \"{0}\" with counting method \"{1}\".".\
            format(self.dataset_to_plot.dataset_to_contour.data_legend,
                   self.dataset_to_plot.counting_method)
