from __future__ import annotations
from typing import Any, TYPE_CHECKING

from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from ..spatial._FM_utils import _plot_map, _plot_vertical_profile

from .._spectral import plot_2dspectrum

if TYPE_CHECKING:
    from ..dataset import DataArray, Dataset


class _DataArrayPlotter:
    """Context aware plotter (sensible plotting according to geometry)."""

    def __init__(self, da: "DataArray") -> None:
        self.da = da

    def __call__(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> Axes:
        """Plot DataArray according to geometry.

        Parameters
        ----------
        ax: matplotlib.axes, optional
            Adding to existing axis, instead of creating new fig
        figsize: (float, float), optional
            specify size of figure
        title: str, optional
            axes title
        **kwargs: Any
            additional arguments passed to the plotting function

        Returns
        -------
        <matplotlib.axes>

        """
        fig, ax = self._get_fig_ax(ax, figsize)

        if self.da.ndim == 1:
            if self.da._has_time_axis:
                return self._timeseries(self.da.values, fig, ax, **kwargs)
            else:
                return self._line_not_timeseries(self.da.values, ax, **kwargs)

        if self.da.ndim == 2:
            return ax.imshow(self.da.values, **kwargs)  # type: ignore

        # if everything else fails, plot histogram
        return self._hist(ax, **kwargs)

    @staticmethod
    def _get_ax(
        ax: Axes | None = None, figsize: tuple[float, float] | None = None
    ) -> Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        return ax

    @staticmethod
    def _get_fig_ax(
        ax: Axes | None = None, figsize: tuple[float, float] | None = None
    ) -> tuple[Figure, Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = plt.gcf()
        return fig, ax

    def hist(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
        **kwargs: Any,
    ) -> Axes:
        """Plot DataArray as histogram (using ax.hist).

        Parameters
        ----------
        bins : int or sequence or str,
            If bins is an integer, it defines the number
            of equal-width bins in the range.
            If bins is a sequence, it defines the bin edges,
            including the left edge of the first bin and the
            right edge of the last bin.
            by default: rcParams["hist.bins"] (default: 10)
        ax: matplotlib.axes, optional
            Adding to existing axis, instead of creating new fig
        figsize: (float, float), optional
            specify size of figure
        title: str, optional
            axes title
        **kwargs: Any
            additional arguments passed to the plotting function

        See Also
        --------
        matplotlib.pyplot.hist

        Returns
        -------
        <matplotlib.axes>

        """
        ax = self._get_ax(ax, figsize)
        if title is not None:
            ax.set_title(title)
        return self._hist(ax, **kwargs)

    def _hist(self, ax: Axes, **kwargs: Any) -> Any:
        result = ax.hist(self.da.values.ravel(), **kwargs)
        ax.set_xlabel(self._label_txt())
        return result

    def line(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> Axes:
        """Plot data as lines (timeseries if time is present)."""
        fig, ax = self._get_fig_ax(ax, figsize)
        if self.da._has_time_axis:
            return self._timeseries(self.da.values, fig, ax, **kwargs)
        else:
            return self._line_not_timeseries(self.da.values, ax, **kwargs)

    def _timeseries(
        self, values: np.ndarray, fig: Figure, ax: Axes, **kwargs: Any
    ) -> Axes:
        if "title" in kwargs:
            title = kwargs.pop("title")
            ax.set_title(title)
        ax.plot(self.da.time, values, **kwargs)
        ax.set_xlabel("time")
        fig.autofmt_xdate()
        ax.set_ylabel(self._label_txt())
        return ax

    def _line_not_timeseries(self, values: np.ndarray, ax: Axes, **kwargs: Any) -> Axes:
        title = kwargs.pop("title") if "title" in kwargs else f"{self.da.time[0]}"
        ax.set_title(title)
        ax.plot(values, **kwargs)
        ax.set_xlabel(self.da.dims[0])
        ax.set_ylabel(self._label_txt())
        return ax

    def _label_txt(self) -> str:
        return f"{self.da.name} [{self.da.unit.short_name}]"

    def _get_first_step_values(self) -> np.ndarray:
        if self.da.n_timesteps > 1:
            return self.da.values[0]
        else:
            return np.squeeze(self.da.values)


class _DataArrayPlotterGrid1D(_DataArrayPlotter):
    """Plot a DataArray with a Grid1D geometry.

    Examples
    --------
    >>> da = mikeio.read("tide1.dfs1")["Level"]
    >>> da.plot()
    >>> da.plot.line()
    >>> da.plot.timeseries()
    >>> da.plot.imshow()
    >>> da.plot.pcolormesh()
    >>> da.plot.hist()

    """

    def __call__(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> Axes:
        _, ax = self._get_fig_ax(ax, figsize)
        if self.da.n_timesteps == 1:
            return self.line(ax, **kwargs)
        else:
            return self.pcolormesh(ax, **kwargs)

    def line(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> Axes:
        """Plot as spatial lines."""
        _, ax = self._get_fig_ax(ax, figsize)
        return self._lines(ax, **kwargs)

    def timeseries(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> Axes:
        """Plot as timeseries."""
        if self.da.n_timesteps == 1:
            raise ValueError("Not possible with single timestep DataArray")
        fig, ax = self._get_fig_ax(ax, figsize)
        return super()._timeseries(self.da.values, fig, ax, **kwargs)

    def imshow(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> Axes:
        """Plot as 2d."""
        if not self.da._has_time_axis:
            raise ValueError(
                "Not possible without time axis. DataArray only has 1 dimension."
            )
        fig, ax = self._get_fig_ax(ax, figsize)
        pos = ax.imshow(self.da.values, **kwargs)
        fig.colorbar(pos, ax=ax, label=self._label_txt())
        return ax

    def pcolormesh(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
        **kwargs: Any,
    ) -> Axes:
        """Plot multiple lines as 2d color plot."""
        if not self.da._has_time_axis:
            raise ValueError(
                "Not possible without time axis. DataArray only has 1 dimension."
            )
        fig, ax = self._get_fig_ax(ax, figsize)
        if title is not None:
            ax.set_title(title)
        pos = ax.pcolormesh(
            self.da.geometry.x,
            self.da.time,
            self.da.values,
            shading="nearest",
            **kwargs,
        )
        _ = fig.colorbar(pos, label=self._label_txt())
        ax.set_xlabel(self.da.geometry._axis_name)
        ax.set_ylabel("time")
        return ax

    def _lines(self, ax: Axes, title: str | None = None, **kwargs: Any) -> Axes:
        """x-lines - one per timestep."""
        if title is not None:
            ax.set_title(title)
        elif self.da.n_timesteps == 1:
            ax.set_title(f"{self.da.time[0]}")
        ax.plot(self.da.geometry.x, self.da.values.T, **kwargs)
        ax.set_xlabel(self.da.geometry._axis_name)
        ax.set_ylabel(self._label_txt())
        return ax


class _DataArrayPlotterGrid2D(_DataArrayPlotter):
    """Plot a DataArray with a Grid2D geometry.

    If DataArray has multiple time steps, the first step will be plotted.

    Examples
    --------
    ```{python}
    import mikeio
    da = mikeio.read("../data/gebco_sound.dfs2")["Elevation"]
    da.plot()
    ```

    """

    def __call__(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> Axes:
        return self.pcolormesh(ax, figsize, **kwargs)

    def contour(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
        **kwargs: Any,
    ) -> Axes:
        """Plot data as contour lines.

        Examples
        --------
        ```{python}
        da = mikeio.read("../data/gebco_sound.dfs2")["Elevation"]
        da.plot.contour()
        ```

        """
        _, ax = self._get_fig_ax(ax, figsize)

        x, y = self._get_x_y()
        values = self._get_first_step_values()

        pos = ax.contour(x, y, values, **kwargs)
        # fig.colorbar(pos, label=self._label_txt())
        ax.clabel(pos, fmt="%1.2f", inline=1, fontsize=9)
        self._set_aspect_and_labels(ax, self.da.geometry, y)
        if title is not None:
            ax.set_title(title)
        return ax

    def contourf(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
        label: str | None = None,
        **kwargs: Any,
    ) -> Axes:
        """Plot data as filled contours.

        Examples
        --------
        ```{python}
        da = mikeio.read("../data/gebco_sound.dfs2")["Elevation"]
        da.plot.contourf()
        ```

        """
        fig, ax = self._get_fig_ax(ax, figsize)

        x, y = self._get_x_y()
        values = self._get_first_step_values()

        label = label if label is not None else self._label_txt()

        pos = ax.contourf(x, y, values, **kwargs)
        fig.colorbar(pos, label=label, pad=0.01)
        self._set_aspect_and_labels(ax, self.da.geometry, y)
        if title is not None:
            ax.set_title(title)
        return ax

    def pcolormesh(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
        label: str | None = None,
        **kwargs: Any,
    ) -> Axes:
        """Plot data as coloured patches.

        Examples
        --------
        ```{python}
        da = mikeio.read("../data/gebco_sound.dfs2")["Elevation"]
        da.plot.pcolormesh()
        ```

        """
        fig, ax = self._get_fig_ax(ax, figsize)

        xn, yn = self._get_xn_yn()
        values = self._get_first_step_values()

        label = label if label is not None else self._label_txt()

        pos = ax.pcolormesh(xn, yn, values, **kwargs)
        fig.colorbar(pos, label=label, pad=0.01)
        self._set_aspect_and_labels(ax, self.da.geometry, yn)
        if title is not None:
            ax.set_title(title)
        return ax

    def _get_x_y(self) -> tuple[np.ndarray, np.ndarray]:
        x = self.da.geometry.x
        y = self.da.geometry.y
        return x, y

    def _get_xn_yn(self) -> tuple[np.ndarray, np.ndarray]:
        xn = self.da.geometry._centers_to_nodes(self.da.geometry.x)
        yn = self.da.geometry._centers_to_nodes(self.da.geometry.y)
        return xn, yn

    @staticmethod
    def _set_aspect_and_labels(ax: Axes, geometry: Any, y: np.ndarray) -> None:
        if geometry.is_spectral:
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel("Directions [degree]")
        elif geometry._is_rotated:
            ax.set_xlabel("[m]")
            ax.set_ylabel("[m]")
        elif geometry.projection == "NON-UTM":
            ax.set_xlabel("[m]")
            ax.set_ylabel("[m]")
        elif geometry.is_geo:
            ax.set_xlabel("Longitude [degrees]")
            ax.set_ylabel("Latitude [degrees]")
            mean_lat = np.mean(y)
            aspect_ratio = 1.0 / np.cos(np.pi * mean_lat / 180)
            ax.set_aspect(aspect_ratio)
        else:
            ax.set_xlabel("Easting [m]")
            ax.set_ylabel("Northing [m]")
            ax.set_aspect("equal")


class _DataArrayPlotterFM(_DataArrayPlotter):
    """Plot a DataArray with a GeometryFM geometry.

    If DataArray has multiple time steps, the first step will be plotted.

    If DataArray is 3D the surface layer will be plotted.

    Examples
    --------
    ```{python}
    import mikeio
    da = mikeio.read("../data/HD2D.dfsu")["Surface elevation"]
    da.plot()
    ```

    """

    def __call__(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> Axes:
        """Plot data as coloured patches."""
        ax = self._get_ax(ax, figsize)
        return self._plot_FM_map(ax, **kwargs)

    def patch(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> Axes:
        """Plot data as coloured patches.

        Examples
        --------
        ```{python}
        da = mikeio.read("../data/HD2D.dfsu")["Surface elevation"]
        da.plot.patch()
        ```

        """
        ax = self._get_ax(ax, figsize)
        kwargs["plot_type"] = "patch"
        return self._plot_FM_map(ax, **kwargs)

    def contour(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> Axes:
        """Plot data as contour lines.

        Examples
        --------
        ```{python}
        da = mikeio.read("../data/HD2D.dfsu")["Surface elevation"]
        da.plot.contour()
        ```

        """
        ax = self._get_ax(ax, figsize)
        kwargs["plot_type"] = "contour"
        return self._plot_FM_map(ax, **kwargs)

    def contourf(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> Axes:
        """Plot data as filled contours.

        Examples
        --------
        ```{python}
        da = mikeio.read("../data/HD2D.dfsu")["Surface elevation"]
        da.plot.contourf()
        ```

        """
        ax = self._get_ax(ax, figsize)
        kwargs["plot_type"] = "contourf"
        return self._plot_FM_map(ax, **kwargs)

    def mesh(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> Axes:
        """Plot mesh only.

        Examples
        --------
        ```{python}
        da = mikeio.read("../data/HD2D.dfsu")["Surface elevation"]
        da.plot.mesh()
        ```

        """
        return self.da.geometry.plot.mesh(figsize=figsize, ax=ax, **kwargs)

    def outline(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> Axes:
        """Plot domain outline (using the boundary_polylines property).

        Examples
        --------
        ```{python}
        da = mikeio.read("../data/HD2D.dfsu")["Surface elevation"]
        da.plot.outline()
        ```

        """
        return self.da.geometry.plot.outline(figsize=figsize, ax=ax, **kwargs)

    def _plot_FM_map(self, ax: Axes, **kwargs: Any) -> Axes:
        values = self._get_first_step_values()

        title = f"{self.da.time[0]}"
        if self.da.geometry.is_layered:
            # select surface as default plotting for 3d files
            values = values[self.da.geometry.top_elements]
            geometry = self.da.geometry.geometry2d
            title = "Surface, " + title
        else:
            geometry = self.da.geometry

        if "label" not in kwargs:
            kwargs["label"] = self._label_txt()
        if "title" not in kwargs:
            kwargs["title"] = title

        return _plot_map(
            node_coordinates=geometry.node_coordinates,
            element_table=geometry.element_table,
            element_coordinates=geometry.element_coordinates,
            boundary_polygon=geometry.boundary_polygon,
            projection=geometry.projection,
            z=values,
            ax=ax,
            **kwargs,
        )


class _DataArrayPlotterFMVerticalColumn(_DataArrayPlotter):
    """Plot a DataArray with a GeometryFMVerticalColumn geometry.

    If DataArray has multiple time steps, the first step will be plotted.

    Examples
    --------
    ```{python}
    import mikeio
    ds = mikeio.read("../data/oresund_sigma_z.dfsu")
    dsp = ds.sel(x=333934.1, y=6158101.5)
    da = dsp["Temperature"]
    da.plot()
    ```
    ```{python}
    da.plot(extrapolate=False, marker='o')
    ```
    ```{python}
    da.plot.pcolormesh()
    ```
    ```{python}
    da.plot.hist()
    ```

    """

    def __call__(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> Axes:
        ax = self._get_ax(ax, figsize)
        return self.line(ax, **kwargs)

    def line(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        extrapolate: bool = True,
        **kwargs: Any,
    ) -> Axes:
        """Plot data as vertical lines."""
        ax = self._get_ax(ax, figsize)
        return self._line(ax, extrapolate=extrapolate, **kwargs)

    def _line(
        self,
        ax: Axes,
        show_legend: bool | None = None,
        extrapolate: bool = True,
        **kwargs: Any,
    ) -> Axes:
        import matplotlib.pyplot as plt

        if "title" in kwargs:
            title = kwargs.pop("title")
            ax.set_title(title)

        if show_legend is None:
            show_legend = len(self.da.time) < 10

        values = self.da.to_numpy()
        zn = self.da._zn
        if extrapolate:
            ze = self.da.geometry._calc_zee(zn)
            values = self.da.geometry._interp_values(zn, values, ze)
        else:
            ze = self.da.geometry.calc_ze(zn)

        ax.plot(values.T, ze.T, label=self.da.time, **kwargs)

        ax.set_xlabel(self._label_txt())
        ax.set_ylabel("z")

        if show_legend:
            plt.legend()

        return ax

    def pcolormesh(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
        **kwargs: Any,
    ) -> Axes:
        """Plot data as coloured patches."""
        fig, ax = self._get_fig_ax(ax, figsize)
        ze = self.da.geometry.calc_ze()
        pos = ax.pcolormesh(
            self.da.time,
            ze,
            self.da.values.T,
            shading="nearest",
            **kwargs,
        )
        fig.colorbar(pos, label=self._label_txt())
        ax.set_xlabel("time")
        fig.autofmt_xdate()
        ax.set_ylabel("z (static)")
        if title is not None:
            ax.set_title(title)
        return ax


class _DataArrayPlotterFMVerticalProfile(_DataArrayPlotter):
    """Plot a DataArray with a 2DV GeometryFMVerticalProfile geometry.

    If DataArray has multiple time steps, the first step will be plotted.

    Examples
    --------
    ```{python}
    import mikeio
    da = mikeio.read("../data/oresund_vertical_slice.dfsu")["Temperature"]
    da.plot()
    ```
    ```{python}
    da.plot.hist()
    ```

    """

    def __call__(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> Axes:
        ax = self._get_ax(ax, figsize)
        return self._plot_transect(ax=ax, **kwargs)

    def _plot_transect(self, **kwargs: Any) -> Axes:
        if "label" not in kwargs:
            kwargs["label"] = self._label_txt()
        if "title" not in kwargs:
            kwargs["title"] = self.da.time[0]
        assert self.da._zn is not None
        if self.da.n_timesteps > 1:
            values = self.da.values[0]
            zn = self.da._zn[0]
        else:
            values = np.squeeze(self.da.values)
            zn = np.squeeze(self.da._zn)  # type: ignore
        g = self.da.geometry
        return _plot_vertical_profile(
            node_coordinates=g.node_coordinates,
            element_table=g.element_table,
            values=values,
            zn=zn,
            is_geo=g.is_geo,
            **kwargs,
        )


class _DataArrayPlotterPointSpectrum(_DataArrayPlotter):
    def __call__(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> Axes:
        # ax = self._get_ax(ax, figsize)
        if self.da.n_frequencies > 0 and self.da.n_directions > 0:
            return self._plot_2dspectrum(figsize=figsize, **kwargs)
        elif self.da.n_frequencies == 0:
            return self._plot_dirspectrum(ax=ax, figsize=figsize, **kwargs)
        elif self.da.n_directions == 0:
            return self._plot_freqspectrum(ax=ax, figsize=figsize, **kwargs)
        else:
            raise ValueError("Spectrum could not be plotted")

    def patch(self, **kwargs: Any) -> Axes:
        kwargs["plot_type"] = "patch"
        return self._plot_2dspectrum(**kwargs)

    def contour(self, **kwargs: Any) -> Axes:
        kwargs["plot_type"] = "contour"
        return self._plot_2dspectrum(**kwargs)

    def contourf(self, **kwargs: Any) -> Axes:
        kwargs["plot_type"] = "contourf"
        return self._plot_2dspectrum(**kwargs)

    def _plot_freqspectrum(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> Axes:
        ax = self._plot_1dspectrum(self.da.frequencies, ax, figsize, **kwargs)  # type: ignore
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel("directionally integrated energy [m*m*s]")
        return ax

    def _plot_dirspectrum(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> Axes:
        ax = self._plot_1dspectrum(self.da.directions, ax, figsize, **kwargs)  # type: ignore
        ax.set_xlabel("directions [degrees]")
        ax.set_ylabel("directional spectral energy [m*m*s]")
        ax.set_xticks(self.da.directions[::2])  # type: ignore
        return ax

    def _plot_1dspectrum(
        self,
        x_values: np.ndarray,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> Axes:
        ax = self._get_ax(ax, figsize)
        y_values = self._get_first_step_values()

        if "linestyle" not in kwargs:
            kwargs["linestyle"] = "-"
        if "marker" not in kwargs:
            kwargs["marker"] = "."

        title = kwargs.pop("title") if "title" in kwargs else self._get_title()
        ax.set_title(title)

        ax.plot(x_values, y_values, **kwargs)
        return ax

    def _plot_2dspectrum(self, **kwargs: Any) -> Axes:
        values = self._get_first_step_values()

        if "figsize" not in kwargs or kwargs["figsize"] is None:
            kwargs["figsize"] = (7, 7)
        if "label" not in kwargs:
            kwargs["label"] = self._label_txt()
        if "title" not in kwargs:
            kwargs["title"] = self._get_title()

        return plot_2dspectrum(
            values,
            frequencies=self.da.geometry.frequencies,
            directions=self.da.geometry.directions,
            **kwargs,
        )

    def _get_title(self) -> str:
        txt = f"{self.da.time[0]}"
        x, y = self.da.geometry.x, self.da.geometry.y
        if x is not None and y is not None:
            if np.abs(x) < 400 and np.abs(y) < 90:
                txt = txt + f", (x, y) = ({x:.5f}, {y:.5f})"
            else:
                txt = txt + f", (x, y) = ({x:.1f}, {y:.1f})"
        return txt


class _DataArrayPlotterLineSpectrum(_DataArrayPlotterGrid1D):
    def __init__(self, da: DataArray) -> None:
        if da.n_timesteps > 1:
            Hm0 = da[0].to_Hm0()
        else:
            Hm0 = da.to_Hm0()
        super().__init__(Hm0)


class _DataArrayPlotterAreaSpectrum(_DataArrayPlotterFM):
    def __init__(self, da: DataArray) -> None:
        if da.n_timesteps > 1:
            Hm0 = da[0].to_Hm0()
        else:
            Hm0 = da.to_Hm0()
        super().__init__(Hm0)


class _DatasetPlotter:
    """Class for plotting scatter plots from datasets."""

    def __init__(self, ds: Dataset) -> None:
        self.ds = ds

    def __call__(
        self, figsize: tuple[float, float] | None = None, **kwargs: Any
    ) -> Axes:
        """Plot multiple DataArrays as time series (only possible dfs0-type data)."""
        if self.ds.dims == ("time",):
            df = self.ds.to_dataframe()
            return df.plot(figsize=figsize, **kwargs)
        else:
            raise ValueError(
                "Could not plot Dataset. Try plotting one of its DataArrays instead..."
            )

    @staticmethod
    def _get_fig_ax(
        ax: Axes | None = None, figsize: tuple[float, float] | None = None
    ) -> tuple[Figure, Axes]:
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = plt.gcf()
        return fig, ax

    def scatter(
        self,
        x: str | int,
        y: str | int,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> Axes:
        """Plot data from two DataArrays against each other in a scatter plot.

        Parameters
        ----------
        x : str or int
            Identifier for first DataArray
        y : str or int
            Identifier for second DataArray
        ax: matplotlib.axes, optional
            Adding to existing axis, instead of creating new fig
        figsize: (float, float), optional
            specify size of figure
        title: str, optional
            axes title
        **kwargs: Any
            additional kwargs will be passed to ax.scatter()

        Returns
        -------
        <matplotlib.axes>

        Examples
        --------
        >>> ds = mikeio.read("oresund_sigma_z.dfsu")
        >>> ds.plot.scatter(x="Salinity", y="Temperature", title="S-vs-T")
        >>> ds.plot.scatter(x=0, y=1, figsize=(9,9), marker='*')

        """
        _, ax = self._get_fig_ax(ax, figsize)
        if "title" in kwargs:
            title = kwargs.pop("title")
            ax.set_title(title)
        xval = self.ds[x].values.ravel()
        yval = self.ds[y].values.ravel()
        ax.scatter(xval, yval, **kwargs)

        ax.set_xlabel(self._label_txt(self.ds[x]))
        ax.set_ylabel(self._label_txt(self.ds[y]))
        return ax

    @staticmethod
    def _label_txt(da: DataArray) -> str:
        return f"{da.name} [{da.unit.name}]"


__all__ = [
    "_DataArrayPlotter",
    "_DataArrayPlotterGrid1D",
    "_DataArrayPlotterGrid2D",
    "_DataArrayPlotterFM",
    "_DataArrayPlotterFMVerticalColumn",
    "_DataArrayPlotterFMVerticalProfile",
    "_DataArrayPlotterPointSpectrum",
    "_DataArrayPlotterLineSpectrum",
    "_DataArrayPlotterAreaSpectrum",
    "_DatasetPlotter",
]
