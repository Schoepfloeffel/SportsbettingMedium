"""
DESCRIPTION:
    Wrapper for interactive plot building with the seaborn library and Qt engine.
    The wrapper tries to build plots easily with the most common methods BUT! still gives the possibility to access the matplotlib object directly.
    Subplotting is not covered with directly.
    
INFO:
    The seaborn.objects interface API is used, which is based on matplotlib.
    Please make sure that your IDE supports Qt.
    Backend is by default set to "Qt5Agg" for interactive plot building.
    Refer to https://seaborn.pydata.org/api.html#objects-interface for documentation and more functionalities.
    
NOTE from documentation:
    Using "Plot.on" method provides access to the underlying matplotlib objects, which may be useful for deep customization.
    But it requires a careful attention to the order of operations by which the Plot is specified, compiled, customized, and displayed.
    
"""

import inspect
from abc import ABC, abstractmethod
from functools import wraps
from typing import Callable, List, Tuple

import matplotlib
import matplotlib.collections as mcoll
import matplotlib.legend as mlegend
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn.objects as so
from seaborn._core.scales import Scale

# from matplotlib import style


# ---- Handle the legend case individually
# Reusable function that moves the last legend entry to the axes. Redefines the axes object as parent.
def _move_legend_from_fig_to_axes(
    fig: matplotlib.figure.Figure, axes: matplotlib.axes._axes.Axes
):
    """
    A hacky way to make the legend working better with the seaborn object system by applying the axes as parent instead of the figure.
    Unfortunately, the seaborn object API draws the legend onto the figure instead of the axes. As a result,
    some basic calls are not implemented (e.g. remove legend or move legend) and flexibility is compromised.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure that holds the original legend

    axes : matplotlib.axes._axes.Axes
        The target axes for the legend

    """
    if fig.legends:  # if legend exist and was drawn
        last_added_legend = fig.legends[-1]
        handles = last_added_legend.legend_handles
        labels = [t.get_text() for t in last_added_legend.get_texts()]
        legend_kws = inspect.signature(matplotlib.legend.Legend).parameters
        props = {
            k: v for k, v in last_added_legend.properties().items() if k in legend_kws
        }

        fig.legends = fig.legends[
            :-1
        ]  # delete last entry of the figure, handle will update the figure automatically

        axes.legend(
            handles,
            labels,
            title=props["title"].get_text(),
            bbox_to_anchor=(1.0, 0.5),
        )  # add the legend to the axes


# ---- Render a Plot
# Renderer interface which has a render method for a seaborn object plot
class Renderer(ABC):
    """
    The Renderer interface specifies the single method "render" which will render a plot build with seaborn objects.
    The Renderer engine should declare a backend within the __init__, if not please check your IDE default.

    """

    @abstractmethod
    def render(self, plot: so.Plot, axes: matplotlib.axes._axes.Axes) -> None:
        pass


# Render Engine
class RenderEngineSeaborn(Renderer):
    """
    The Concrete Render class for the seaborn object API.
    QT5AGG is used as backend and the plot.on method is used to get access to more customization.

    """

    def __init__(self) -> None:
        backend = matplotlib.get_backend()
        if backend not in ["Qt5Agg"]:
            print(f"Used {backend} switched to Qt5Agg")
            try:
                matplotlib.use("Qt5Agg")
            except ImportError:
                print(f"Keeping {backend} because Qt5Agg cannot be imported")
            plt.ion()

    def render(self, plot: so.Plot, axes: matplotlib.axes._axes.Axes) -> None:
        """
        This method will draw a seaborn object plot onto a matplotlib axes.
        Before the seaborn object plot will be drawn, all artists that visualize data are removed.
        Thus, data artists Line2D, Collections and Rectangles are not drawn twice. Further data artists could be added in the future.
        Legend is a special case: The seaborn object API always draws the legend onto the figure.
        Internal calls are handled onto the figure legend but the end step involves moving the legend from the figure to the axes object to have more flexibility.

        Parameters
        ----------
        plot : so.Plot
            seaborn object plot which holds the data.
        axes : matplotlib.axes._axes.Axes
            The target axes where the seaborn plot will be drawn on.

        Returns
        -------
        None

        """
        artists = axes.get_children()  # get all axes elements
        for pos, art in enumerate(artists):
            try:
                if isinstance(
                    art,
                    mcoll.Collection
                    | mlines.Line2D
                    | mpatches.Rectangle
                    | mlegend.Legend,
                ):
                    artists[
                        pos
                    ].remove()  # remove all artists that show data points and legend (see https://matplotlib.org/3.7.1/api/artist_api.html) with remove method to trigger the refresh event. Simple deletion does not work here!
            except NotImplementedError:
                artists.pop(pos)
        plot.theme(matplotlib.rcParams).on(
            axes
        ).show()  # render always with theme method! Contextmanager under the hood will set the params for one render cycle.

        _move_legend_from_fig_to_axes(axes.get_figure(), axes)


# ---- Build a Plot
# builder interface with methods the builder needs
class Builder(ABC):
    """
    The Builder interface specifies methods for creating the different parts of
    the Plotting object.

    """

    @abstractmethod
    def add_data(self, df: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def add_xdata(self, xdata: pd.Series) -> None:
        pass

    @abstractmethod
    def add_ydata(self, ydata: pd.Series) -> None:
        pass

    @abstractmethod
    def add_mark(
        self, mark: so.Mark, *transforms: so.Stat | so.Move | so.Mark, **kwargs
    ) -> None:
        pass

    @abstractmethod
    def set_scale(self, **scale: Scale) -> None:
        pass

    @abstractmethod
    def add_title(self, label_str: str) -> None:
        pass

    @abstractmethod
    def set_xlabel(self, label_str: str) -> None:
        pass

    @abstractmethod
    def set_ylabel(self, label_str: str) -> None:
        pass

    @abstractmethod
    def set_xlim(self, limit_tuple: Tuple[int, int]) -> None:
        pass

    @abstractmethod
    def set_ylim(self, limit_tuple: Tuple[int, int]) -> None:
        pass

    @abstractmethod
    def set_xtick(self, ticks: List[float | str]) -> None:
        pass

    @abstractmethod
    def set_ytick(self, ticks: List[float | str]) -> None:
        pass


# builder
class PlotBuilder(Builder):
    """
    The Concrete Builder classes follow the Builder interface and provide
    specific implementations of the building steps.

    """

    def __init__(self, renderer: Renderer = RenderEngineSeaborn()) -> None:
        """
        A fresh builder instance contains a blank plot object, which is
        used in further assembly. The style and renderer is also set in here. A custom style is used from the URL.

        """
        self.is_initialized = False
        matplotlib.style.use(
            "https://raw.githubusercontent.com/Schoepfloeffel/mplstyles/main/schoepfloeffel_style_1.mplstyle"
        )
        self._reset()
        self.plot_renderer = renderer
        self.is_initialized = True
        self.df = pd.DataFrame()

    def _render_with_engine(plot_func: Callable) -> None:
        """
        Method will be used as decorator for methods in the "PlotBuilder" class and will call the render engine
        after the method call to ensure interactive plot building

        Parameters
        ----------
        plot_func : Callable
            The plot builder function/method

        """

        @wraps(plot_func)
        def wrapper(self, *args, **kwargs):
            plot_func(self, *args, **kwargs)
            print(
                f"Rendering seaborn object with method: {plot_func.__name__.lstrip('_').upper()}"
            )
            self.plot_renderer.render(self.p, self.axes)
            return None

        return wrapper

    def _reset(self) -> None:
        """
        Will allocate an empty matplotlib figure, axes and an empty seaborn object plot.
        If it was already allocated within the instance, we just clear the axes call another empty seaborn object plot.

        NOTE:
            The method should be not used outside of the class, but the possiblity is given to the user to call it.

        """

        if not self.is_initialized:
            (
                self.fig,
                self.axes,
            ) = plt.subplots()  # allocate empty figure and empty axes at __init__
        else:
            self.fig.legends = (
                []
            )  # clear legend (legend is always a bit of a special artist in matplotlib/seaborn)
            self.axes.cla()  # clear axes that we wrote on
        self.p = so.Plot()  # init a fresh so.Plot instance
        self.xdata = None
        self.ydata = None

    def _add_data(self, **kwargs) -> None:
        """
        "Private" method to overwrite the data of the seaborn object plot

        """

        self.p = so.Plot(data=self.df, x=self.xdata, y=self.ydata, **kwargs)

    def add_data(self, df: pd.DataFrame, **kwargs):
        """
        Adds a pd.Dataframe to the seaborn object plot. This will not draw the plot!
        Sequential methods will use the data to draw the plots/legends/colors etc.

        Parameters
        ----------
        df : pd.DataFrame
        **kwargs : Any
            Further arguments to add layer specific variables (see https://seaborn.pydata.org/generated/seaborn.objects.Plot.add.html).

        Raises
        ------
        TypeError
            Only pd.Dataframes are allowed to follow a strict convention.

        """

        if isinstance(df, pd.DataFrame):
            self.df = df
        else:
            raise TypeError("Please supply a pd.Datframe as data")
        self._add_data(**kwargs)

    def add_xdata(self, xdata: pd.Series) -> None:
        """
        Adds a pd.Series to the seaborn object plot as data for the x-axis. This will not draw the plot!
        Sequential methods will use the data to draw the plots/legends/colors etc.

        Parameters
        ----------
        xdata : pd.Series
            the pd.Series can be derived from the pd.DataFrame that was added with "add_data" method (e.g. df["myColumn"])

        Raises
        ------
        TypeError
            Only pd.Series are allowed to follow a strict convention.

        """

        if isinstance(xdata, pd.Series):  # if None or empty
            self.xdata = xdata
        else:
            raise TypeError("Please supply a pd.Series for the x-data")
        self._add_data()

    def add_ydata(self, ydata: pd.Series) -> None:
        """
        Adds a pd.Series to the seaborn object plot as data for the y-axis. This will not draw the plot!
        Sequential methods will use the data to draw the plots/legends/colors etc.

        Parameters
        ----------
        ydata : pd.Series
            the pd.Series can be derived from the pd.DataFrame that was added with "add_data" method (e.g. df["myColumn"])

        Raises
        ------
        TypeError
            Only pd.Series are allowed to follow a strict convention.

        """

        if isinstance(ydata, pd.Series):  # if None or empty
            self.ydata = ydata
        else:
            raise TypeError("Please supply a pd.Series for the y-data")
        self._add_data()

    @_render_with_engine
    def add_mark(
        self, mark: so.Mark, *transforms: so.Stat | so.Move | so.Mark, **kwargs
    ) -> None:
        """
        Adds the "heart" of the plot to the seaborn object plot.
        Manipulates the visual representation, the statistical or positional transformation of the data.
        This method will render the data into the plot object, in contrast to add_data methods!

        NOTE:
            Please add data (add_data, add_xdata, add_ydata) to the the plot class before calling this method.
            Please add title, scaling, misc after call this method.
            The method can change position and scaling of the plot.
            e.g.:
                diamonds = sns.load_dataset("diamonds")
                plotter = PlotBuilder(RenderEngineSeaborn())
                plotter.add_data(diamonds)
                plotter.add_xdata(diamonds["carat"])
                plotter.add_ydata(diamonds["price"])
                plotter.add_mark(so.Dots(alpha=0.3, fillalpha=0.3), color='cut')
                plotter.add_mark(so.Line(alpha=0.5, color='#002F2F'), so.PolyFit())
                plotter.set_ylim((0,20000))
                sns.move_legend(plotter.axes, "lower center")

        Parameters
        ----------
        mark : so.Mark
            Visual representation of the seaborn plot.
        *transforms : so.Stat | so.Move | so.Mark
            Seaborn objects for statistical, positional or visual transformations.
        **kwargs : Any
            Further arguments to add layer specific variables (see https://seaborn.pydata.org/generated/seaborn.objects.Plot.add.html).

        Raises
        ------
        TypeError
            Only specified seaborn objects are allowed in mark and *transforms (see method).

        """
        if not isinstance(mark, so.Mark):
            raise TypeError(f"Expected so.Mark from seaborn object, got {type(mark)}")
        if not all(
            isinstance(transform, so.Stat | so.Move | so.Mark)
            for transform in transforms
        ):
            raise TypeError(
                f"Expected so.Mark or so.Stat from seaborn object, got {list(map(lambda x: type(x), transforms))}"
            )
        self.p = self.p.add(mark, *transforms, **kwargs)

    @_render_with_engine
    def set_scale(self, **scale: Scale) -> None:
        """
        Adds a scale to the current mark. This will change the visual properties (e.g., ticks, scales, data points)
        See https://seaborn.pydata.org/generated/seaborn.objects.Plot.scale for more information and usage

        Parameters
        ----------
        **scale : Scale
            Base class for objects that map data values to visual properties
            (e.g., color=so.Nominal(["#008fd5", "#fc4f30", "#e5ae38"] will pass nominal scale to the colors of the mark object))

        """

        self.p = self.p.scale(**scale)

    @_render_with_engine
    def add_title(self, label_str: str) -> None:
        """
        Adds title to the seaborn object plot.

        Parameters
        ----------
        label_str : str

        """

        self.p = self.p.label(title=label_str)

    @_render_with_engine
    def set_xlabel(self, label_str: str) -> None:
        """
        Adds x label to the seaborn object plot.

        Parameters
        ----------
        label_str : str

        """

        self.p = self.p.label(x=label_str)

    @_render_with_engine
    def set_ylabel(self, label_str: str) -> None:
        """
        Adds y label to the seaborn object plot.

        Parameters
        ----------
        label_str : str

        """

        self.p = self.p.label(y=label_str)

    @_render_with_engine
    def set_xlim(self, limit_tuple: Tuple[int, int]) -> None:
        """
        Sets scale limits of the x-axis seaborn object plot.

        Parameters
        ----------
        limit_tuple : Tuple[int, int]

        """

        self.p = self.p.limit(x=limit_tuple)

    @_render_with_engine
    def set_ylim(self, limit_tuple: Tuple[int, int]) -> None:
        """
        Sets scale limits of the y-axis seaborn object plot.

        Parameters
        ----------
        limit_tuple : Tuple[int, int]

        """

        self.p = self.p.limit(y=limit_tuple)

    def set_xtick(self, ticks: List[float | str]) -> None:
        """
        Sets ticks of the x-axis on matplotlib axes (not on seaborn plot object. This can be cumbersome, see https://seaborn.pydata.org/generated/seaborn.objects.Continuous).
        If you want to use the seaborn plot object fot ticks, check the "scale" method: https://seaborn.pydata.org/generated/seaborn.objects.Plot.scale.html

        Parameters
        ----------
        ticks : List[float | str]

        """

        self.axes.set_xticks(ticks)

    def set_ytick(self, ticks: List[float | str]) -> None:
        """
        Sets ticks of the y-axis on matplotlib axes (not on seaborn plot object. This can be cumbersome, see https://seaborn.pydata.org/generated/seaborn.objects.Continuous).
        If you want to use the seaborn plot object fot ticks, check the "scale" method: https://seaborn.pydata.org/generated/seaborn.objects.Plot.scale.html

        Parameters
        ----------
        ticks : List[float | str]

        """

        self.axes.set_yticks(ticks)

    def __del__(self) -> None:
        """
        Destructor method. Closes the open figure if the reference is destroyed.

        """

        plt.close(self.fig)
