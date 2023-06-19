from abc import ABC, abstractmethod
import numpy as np
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from matplotlib import colors
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
        self.color_axes = {}  # { plot : color_axis } pair
        self.color_bar = {}
        self.collections = {}  # { plot : color_axis } pair
        self.info_text = "" #  additional info text

        self._caxes_origin = [0.6, 0.05]
        self._caxes_current_origin = self._caxes_origin.copy()
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
                raise RuntimeError(f"Unknown projection method {value}.")
        elif isinstance(value, ProjectionBase):
            self._projection = value
        else:
            raise RuntimeError(f"Unknown projection input {type(value)}")

    def clear(self):
        """
        Remove all plots that are connected to the stereonet
        and redraw the primitive circle
        """
        for _, caxis in self.color_axes.items():
            if caxis is not None:
                caxis.remove()

        self.info_text = ""
        self._caxes_current_origin = self._caxes_origin.copy()
        self.color_axes.clear()
        self.color_bar.clear()

        # for _, collection in self.collections.items():
        #     collection.remove()
        self.collections.clear()

        self.plots.clear()
        self.data_axes.clear()
        self.data_axes.set_axis_off()
        self.data_axes.set_aspect(
            aspect="equal", adjustable=None, anchor="W")

        self.draw_primitive_circle()

    def append_plot(self, plot):
        """
        Append one data plot into the current plot.
        If there is color info in the plot, request a color axis.
        """
        timer = Timer()
        timer.start()

        self.plots.append(plot)

        # register this data
        if plot in self.collections:
            raise RuntimeError(
                f"Plot with legend {plot.data_legend} has already been added.")

        collection = plot.draw()

        if plot.dataset_to_plot.color_data is None:
            # color axis is None
            self.color_axes[plot] = None
            self.color_bar[plot] = None
        else:
            caxis = self._create_next_color_axis()
            cb = plt.colorbar(collection, cax=caxis,
                              orientation="vertical", spacing="proportional")
            cb.formatter.set_powerlimits((0, 0))
            cb.set_label(plot.dataset_to_plot.color_legend)
            caxis.yaxis.set_label_position("left")
            self.color_axes[plot] = caxis
            self.color_bar[plot] = cb

        self.collections[plot] = collection

        timer.stop()
        log_info(
            f"{type(plot).__name__} added to stereonet "
            f"with options:{plot.plot_options}")
        return collection


    def generate_plots(self, show_plot=True):
        """
        Plot all datasets on this stereonet
        """
        # draw legends
        self.data_axes.legend(loc="upper right")
        log_info("All plots are successfully generated")

        # info string
        info_txt = self.info_text + '\n'
        for plot in self.plots:
            info_txt += plot.info_text() + '\n'
        self.data_axes.text(1.05, 0.95, info_txt,
                            transform=self.data_axes.transAxes,
                            **info_font)

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
        self.data_axes.scatter(x_cross, y_cross, s=100,
                               color="grey", marker='+')
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
        self._plot_options = {}
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
        raise RuntimeError(f"{key} is not in options {opt_copy}")


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
        cmap_norm = None
        x, y = self.stereonet.project(self.dataset_to_plot).T

        # extract color data, if we have them:
        if self.dataset_to_plot.color_data is not None:
            if "color" in opt:
                opt.pop("color")

            cmap_limits_autoset = False
            if "cmap_limits" in opt:
                cmap_limits = opt.pop("cmap_limits")
                if cmap_limits is None:
                    cmap_limits_autoset = True
            else:
                cmap_limits_autoset = True
            if cmap_limits_autoset:
                cmap_limits = [
                    min(self.dataset_to_plot.color_data),
                    max(self.dataset_to_plot.color_data)]

            if "cmap_center" in opt:
                cmap_center = opt.pop("cmap_center")
                if cmap_center is not None:
                    cmap_norm = colors.TwoSlopeNorm(
                        cmap_center, cmap_limits[0], cmap_limits[1])

            plot = self.stereonet.data_axes.scatter(
                x, y, c=self.dataset_to_plot.color_data,
                label=self.dataset_to_plot.data_legend,
                norm=cmap_norm, **opt)
        else:
            if "color" not in opt:
                opt["color"] = "black"
            if "cmap_limits" in opt:
                opt.pop("cmap_limits")
            if "cmap_center" in opt:
                opt.pop("cmap_center")
            if "cmap" in opt:
                opt.pop("cmap")

            plot = self.stereonet.data_axes.scatter(x, y,
                                                    label=self.dataset_to_plot.data_legend, **opt)
        return plot

    def info_text(self):
        return "Scatter plot for dataset "\
            f"\"{self.dataset_to_plot.data_legend}\" "\
            f"contains {self.dataset_to_plot.n_entries} points."


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
        return "Big circle plot for dataset " \
            f"\"{self.dataset_to_plot.data_legend}\" " \
            f"contains {self.dataset_to_plot.n_entries} entries."


class ContourPlot(PlotBase):
    """
    Plotting the contour data
    """

    def __init__(self, stereonet, contour_data, **kwargs):
        super().__init__(stereonet, contour_data, **kwargs)
        if not isinstance(contour_data, ContourData):
            raise RuntimeError("Only ContourData is allowed for ContourPlot.")

    def draw(self):
        # make a copy of the options
        opt = dict(self.plot_options).copy()

        X, Y = self.stereonet.project(self.dataset_to_plot.nodes).T
        count = self.dataset_to_plot.color_data
        n = self.extract_option(opt, "n_intervals") + 1

        if "lim" not in opt:
            levels = np.linspace(0, count.max(), n)
        else:
            lim = self.extract_option(opt, "lim")
            if lim is None:
                levels = np.linspace(0, count.max(), n)
            else:
                levels = np.linspace(lim[0], lim[1], n)

        if self.extract_option(opt, "filled"):
            plot = self.stereonet.data_axes.tricontourf(
                X, Y, count, levels, extend="both", **opt)
        else:
            plot = self.stereonet.data_axes.tricontour(
                X, Y, count, levels, **opt)
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
        return "ContourPlot of dataset "\
            f"\"{self.dataset_to_plot.dataset_to_contour.data_legend}\" "\
            f"with counting method \"{self.dataset_to_plot.counting_method}\"."
