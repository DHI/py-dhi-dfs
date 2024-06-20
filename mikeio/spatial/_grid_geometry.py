from __future__ import annotations
from functools import cached_property
from pathlib import Path
import warnings
from typing import Any, Sequence, Tuple, TYPE_CHECKING, List, overload
from dataclasses import dataclass
import numpy as np

from mikecore.Projections import Cartography

from ..exceptions import OutsideModelDomainError

from ._geometry import (
    BoundingBox,
    GeometryPoint2D,
    GeometryPoint3D,
    GeometryUndefined,
    _Geometry,
)

from ..eum import EUMType, ItemInfo

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from ..spatial import GeometryFM2D
    from ..dataset import DataArray


def _check_equidistant(x: np.ndarray) -> None:
    d = np.diff(x)
    if len(d) > 0 and not np.allclose(d, d[0]):
        raise NotImplementedError("values must be equidistant")


def _parse_grid_axis(
    name: str,
    x: np.ndarray | None,
    x0: float = 0.0,
    dx: float | None = None,
    nx: int | None = None,
) -> Tuple[float, float, int]:
    if x is not None:
        x = np.asarray(x)
        _check_equidistant(x)
        if len(x) > 1 and x[0] > x[-1]:
            raise ValueError(f"{name} values must be increasing")
        x0 = x[0]
        dx = x[1] - x[0] if len(x) > 1 else 1.0
        nx = len(x)
    else:
        if nx is None:
            raise ValueError(f"n{name} must be provided")
        if dx is None:
            raise ValueError(f"d{name} must be provided")
        if dx <= 0:
            raise ValueError(f"d{name} must be positive")
    return x0, dx, nx


def _print_axis_txt(name: str, x: np.ndarray, dx: float) -> str:
    n = len(x)
    txt = f"{name}: [{x[0]:0.4g}"
    if n > 1:
        txt = txt + f", {x[1]:0.4g}"
    if n == 3:
        txt = txt + f", {x[2]:0.4g}"
    if n > 3:
        txt = txt + f", ..., {x[-1]:0.4g}"
    txt = txt + f"] (n{name}={n}, d{name}={dx:0.4g})"
    return txt


@dataclass
class Grid1D(_Geometry):
    """1D grid (node-based)
    axis is increasing and equidistant

    Parameters
    ----------
    x : array_like
        node coordinates
    x0 : float
        first node coordinate
    dx : float
        grid spacing
    nx : int
        number of nodes
    projection : str
        projection string
    origin : tuple
        not commonly used
    orientation : float
        not commonly used
    node_coordinates : array_like
        coordinates of nodes in 2D or 3D space
    axis_name : str
        name of axis, by default "x"

    Examples
    --------
    ```{python}
    import mikeio
    mikeio.Grid1D(nx=3,dx=0.1)
    ```

    ```{python}
    mikeio.Grid1D(x=[0.1, 0.5, 0.9])
    ```
    """

    _dx: float
    _nx: int
    _x0: float
    _orientation: float
    _origin: Tuple[float, float]
    _projstr: str

    def __init__(
        self,
        x: np.ndarray | None = None,
        *,
        x0: float = 0.0,
        dx: float | None = None,
        nx: int | None = None,
        projection: str = "NON-UTM",
        origin: Tuple[float, float] = (0.0, 0.0),
        orientation: float = 0.0,
        node_coordinates: np.ndarray | None = None,
        axis_name: str = "x",
    ):
        """Create equidistant 1D spatial geometry"""
        super().__init__(projection=projection)
        self._origin = (0.0, 0.0) if origin is None else (origin[0], origin[1])
        assert len(self._origin) == 2, "origin must be a tuple of length 2"
        self._orientation = orientation
        self._x0, self._dx, self._nx = _parse_grid_axis("x", x, x0, dx, nx)

        if node_coordinates is not None and len(node_coordinates) != self.nx:
            raise ValueError("Length of node_coordinates must be n")
        self._nc = node_coordinates

        self._axis_name = axis_name

    @property
    def ndim(self) -> int:
        return 1

    @property
    def default_dims(self) -> Tuple[str, ...]:
        return ("x",)

    def __repr__(self) -> str:
        out = ["<mikeio.Grid1D>", _print_axis_txt("x", self.x, self.dx)]
        return "\n".join(out)

    def __str__(self) -> str:
        return f"Grid1D (n={self.nx}, dx={self.dx:.4g})"

    def find_index(self, x: float, **kwargs: Any) -> int:
        """Find nearest point

        Parameters
        ----------
        x : float
            x-coordinate of point

        Returns
        -------
        int
            index of nearest point
        **kwargs : Any
            Not used

        See Also
        --------
        [](`mikeio.Dataset.sel`)
        """

        d = (self.x - x) ** 2

        return int(np.argmin(d))

    def get_spatial_interpolant(
        self, coords: Tuple[np.ndarray, np.ndarray], **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = coords[0][0]  # TODO accept list of points

        assert self.nx > 1, "Interpolation not possible for Grid1D with one point"
        d = np.abs(self.x - x)
        ids = np.argsort(d)[0:2]

        if x > self.x.max() or x < self.x.min():
            weights = np.array([np.nan, np.nan])
        else:
            weights = 1 - d[ids] / d[ids].sum()
            assert np.allclose(weights.sum(), 1.0)
        assert len(ids) == 2
        assert len(weights) == 2
        return ids, weights

    def interp(self, data: np.ndarray, ids: np.ndarray, weights: np.ndarray) -> Any:
        return np.dot(data[:, ids], weights)

    @property
    def dx(self) -> float:
        """grid spacing"""
        return self._dx

    @property
    def x(self) -> np.ndarray:
        """array of node coordinates"""
        x1 = self._x0 + self.dx * (self.nx - 1)
        return np.linspace(self._x0, x1, self.nx)

    @property
    def nx(self) -> int:
        """number of grid points"""
        return self._nx

    @property
    def origin(self) -> Tuple[float, float]:
        return self._origin

    @property
    def orientation(self) -> float:
        return self._orientation

    def isel(
        self, idx: int | np.int64 | slice, axis: int | None = None
    ) -> "Grid1D" | GeometryPoint2D | GeometryPoint3D | GeometryUndefined:
        """Get a subset geometry from this geometry

        Parameters
        ----------
        idx : int or slice
            index or slice
        axis : int, optional
            Not used for Grid1D, by default None

        Returns
        -------
        GeometryPoint2D or GeometryPoint3D or GeometryUndefined
            The geometry of the selected point

        Examples
        --------
        ```{python}
        import mikeio
        g = mikeio.Grid1D(nx=3,dx=0.1)
        g
        ```

        ```{python}
        g.isel([1,2])
        ```

        ```{python}
        g.isel(1)
        ```
        """

        if not np.isscalar(idx):
            nc = None if self._nc is None else self._nc[idx, :]
            return Grid1D(
                x=self.x[idx],
                projection=self.projection,
                origin=self.origin,
                orientation=self.orientation,
                node_coordinates=nc,
            )

        if self._nc is None:
            return GeometryUndefined()
        else:
            coords = self._nc[idx, :]  # type: ignore
            if len(coords) == 3:
                x, y, z = coords
                return GeometryPoint3D(x=x, y=y, z=z, projection=self.projection)
            else:
                x, y = coords
                return GeometryPoint2D(x=x, y=y, projection=self.projection)


class _Grid2DPlotter:
    """Plot Grid2D

    Examples
    --------
    ```{python}
    import mikeio
    g = mikeio.read("../data/waves.dfs2").geometry
    ax = g.plot()
    ```
    """

    def __init__(self, geometry: "Grid2D") -> None:
        self.g = geometry

    def __call__(
        self,
        ax: Axes | None = None,
        figsize: Tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Plot bathymetry as coloured patches"""
        ax = self._get_ax(ax, figsize)
        return self._plot_grid(ax, **kwargs)

    @staticmethod
    def _get_ax(
        ax: Axes | None = None, figsize: Tuple[float, float] | None = None
    ) -> Axes:
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        return ax

    def _plot_grid(
        self,
        ax: Axes,
        title: str | None = None,
        color: str = "0.5",
        linewidth: float = 0.6,
        **kwargs: Any,
    ) -> Axes:
        g = self.g
        xn = g._centers_to_nodes(g.x)
        yn = g._centers_to_nodes(g.y)
        for yj in yn:
            ax.plot(
                xn, yj * np.ones_like(xn), color=color, linewidth=linewidth, **kwargs
            )
        for xj in xn:
            ax.plot(
                xj * np.ones_like(yn), yn, color=color, linewidth=linewidth, **kwargs
            )
        if title is not None:
            ax.set_title(title)
        self._set_aspect_and_labels(ax)
        return ax

    def outline(
        self,
        title: str = "Outline",
        ax: Axes | None = None,
        figsize: Tuple[float, float] | None = None,
        color: str = "0.4",
        linewidth: float = 1.2,
        **kwargs: Any,
    ) -> Axes:
        """Plot Grid2D outline

        Examples
        --------
        ```{python}
        g = mikeio.read("../data/waves.dfs2").geometry
        g.plot.outline()
        ```
        """
        ax = self._get_ax(ax, figsize)
        g = self.g
        xn = g._centers_to_nodes(g.x)
        yn = g._centers_to_nodes(g.y)
        for yj in [yn[0], yn[-1]]:
            ax.plot(
                xn, yj * np.ones_like(xn), color=color, linewidth=linewidth, **kwargs
            )
        for xj in [xn[0], xn[-1]]:
            ax.plot(
                xj * np.ones_like(yn), yn, color=color, linewidth=linewidth, **kwargs
            )
        if title is not None:
            ax.set_title(title)
        self._set_aspect_and_labels(ax)

        return ax

    def _set_aspect_and_labels(self, ax: Axes) -> None:
        g = self.g
        if g.is_spectral:
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel("Directions [degree]")
        elif g._is_rotated:
            ax.set_xlabel("[m]")
            ax.set_ylabel("[m]")
        elif g.projection == "NON-UTM":
            ax.set_xlabel("[m]")
            ax.set_ylabel("[m]")
        elif g.is_geo:
            ax.set_xlabel("Longitude [degrees]")
            ax.set_ylabel("Latitude [degrees]")
            mean_lat = np.mean(g.y)
            aspect_ratio = 1.0 / np.cos(np.pi * mean_lat / 180)
            ax.set_aspect(aspect_ratio)
        else:
            ax.set_xlabel("Easting [m]")
            ax.set_ylabel("Northing [m]")
            ax.set_aspect("equal")


@dataclass
class Grid2D(_Geometry):
    """2D grid
    Origin in the center of cell in lower-left corner
    x and y axes are increasing and equidistant
    """

    _dx: float
    _nx: int
    _x0: float
    _dy: float
    _ny: int
    _y0: float
    _projstr: str
    _origin: Tuple[float, float]
    _orientation: float
    is_spectral: bool
    bathymetry: DataArray | None = None

    def __init__(
        self,
        *,
        x: np.ndarray | None = None,
        x0: float = 0.0,
        dx: float | None = None,
        nx: int | None = None,
        y: np.ndarray | None = None,
        y0: float = 0.0,
        dy: float | None = None,
        ny: int | None = None,
        bbox: Tuple[float, float, float, float] | None = None,
        projection: str = "LONG/LAT",
        origin: Tuple[float, float] | None = None,
        orientation: float = 0.0,
        axis_names: Tuple[str, str] = ("x", "y"),
        is_spectral: bool = False,
        is_vertical: bool = False,
        bathymetry: np.ndarray | None = None,
    ):
        """Create equidistant 2D spatial geometry

        Parameters
        ----------
        x : array_like, optional
            x coordinates of cell centers
        x0 : float, optional
            x coordinate of lower-left corner of first cell
        dx : float, optional
            x cell size
        nx : int, optional
            number of cells in x direction
        y : array_like, optional
            y coordinates of cell centers
        y0 : float, optional
            y coordinate of lower-left corner of first cell
        dy : float, optional
            y cell size
        ny : int, optional
            number of cells in y direction
        bbox : tuple, optional
            (x0, y0, x1, y1) of bounding box
        projection : str, optional
            projection string, by default "NON-UTM"
        origin : tuple, optional
            user-defined origin, by default None
        orientation : float, optional
            rotation angle in degrees, by default 0.0
        axis_names : tuple, optional
            names of x and y axes, by default ("x", "y")
        is_spectral : bool, optional
            if True, the grid is spectral, by default False
        is_vertical : bool, optional
            if True, the grid is vertical, by default False
        bathymetry : array_like, optional
            bathymetry data, by default None

        Examples
        --------
        ```{python}
        import mikeio
        mikeio.Grid2D(x0=12.0, nx=2, dx=0.25, y0=55.0, ny=3, dy=0.25, projection="LONG/LAT")
        ```
        """
        super().__init__(projection=projection)
        self._shift_origin_on_write = origin is None  # user-constructed
        self._origin = (0.0, 0.0) if origin is None else (origin[0], origin[1])
        assert len(self._origin) == 2, "origin must be a tuple of length 2"
        self._orientation = orientation
        self.__xx = None
        self.__yy = None

        self._axis_names = axis_names

        if bbox is not None:
            if (x0 != 0.0) or (y0 != 0.0):
                raise ValueError("x0,y0 cannot be provided together with bbox")
            self._create_in_bbox(bbox, dx=dx, dy=dy, nx=nx, ny=ny)
        else:
            self._x0, self._dx, self._nx = _parse_grid_axis("x", x, x0, dx, nx)
            dy = self._dx if dy is None else dy
            self._y0, self._dy, self._ny = _parse_grid_axis("y", y, y0, dy, ny)

        self.is_spectral = is_spectral
        self.is_vertical = is_vertical

        if bathymetry is not None:
            from ..dataset import DataArray

            bathy_da = DataArray(
                data=bathymetry,
                geometry=self,
                item=ItemInfo(name="Bathymetry", itemtype=EUMType.Bathymetry),
            )
            self.bathymetry = bathy_da
        else:
            self.bathymetry = None

        self.plot = _Grid2DPlotter(self)

    @property
    def default_dims(self) -> Tuple[str, ...]:
        return ("y", "x")

    @property
    def ndim(self) -> int:
        return 2

    @property
    def _is_rotated(self) -> Any:
        return np.abs(self._orientation) > 1e-5

    def _create_in_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        dx: float | Tuple[float, float] | None = None,
        dy: float | None = None,
        nx: int | None = None,
        ny: int | None = None,
    ) -> None:
        """create 2d grid in bounding box, specifying spacing or shape

        Parameters
        ----------
        bbox : array(float)
            [left, bottom, right, top] =
            [x0-dx/2, y0-dy/2, x1+dx/2, y1+dy/2]
        dx : float, optional
            grid resolution in x-direction,
            will overwrite left and right if given
        dy : float, optional
            grid resolution in y-direction,
            will overwrite bottom and top if given
        nx : int, optional
            number of points in x-direction can be None,
            in which case the value will be inferred
        ny : int, optional
            number of points in y-direction can be None,
            in which case the value will be inferred
        """
        left, bottom, right, top = self._parse_bbox(bbox)

        xr = right - left  # dx too large
        yr = top - bottom  # dy too large

        if (dx is None and dy is None) and (nx is None and ny is None):
            if xr <= yr:
                nx = 10
                ny = int(np.ceil(nx * yr / xr))
            else:
                ny = 10
                nx = int(np.ceil(ny * xr / yr))
        if nx is None and ny is not None:
            nx = int(np.ceil(ny * xr / yr))
        if ny is None and nx is not None:
            ny = int(np.ceil(nx * yr / xr))
        if isinstance(dx, tuple):
            dx, dy = dx
        dy = dx if dy is None else dy

        self._x0, self._dx, self._nx = self._create_in_bbox_1d("x", left, right, dx, nx)
        self._y0, self._dy, self._ny = self._create_in_bbox_1d("y", bottom, top, dy, ny)

    @staticmethod
    def _parse_bbox(
        bbox: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        left = bbox[0]
        bottom = bbox[1]
        right = bbox[2]
        top = bbox[3]

        if left > right:
            raise (
                ValueError(
                    f"Invalid x axis, left: {left} must be smaller than right: {right}"
                )
            )

        if bottom > top:
            raise (
                ValueError(
                    f"Invalid y axis, bottom: {bottom} must be smaller than top: {top}"
                )
            )
        return left, bottom, right, top

    @staticmethod
    def _create_in_bbox_1d(
        axname: str,
        left: float,
        right: float,
        dx: float | None = None,
        nx: int | None = None,
    ) -> Tuple[float, float, int]:
        xr = right - left
        if dx is not None:
            nx = int(np.ceil(xr / dx))

            # overwrite left and right! to make dx fit
            xcenter = left + xr / 2
            left = xcenter - dx * nx / 2
            right = xcenter + dx * nx / 2
        elif nx is not None:
            dx = xr / nx
        else:
            raise ValueError(f"Provide either d{axname} or n{axname}")
        x0 = left + dx / 2
        return x0, dx, nx

    def __repr__(self) -> str:
        out = (
            ["<mikeio.Grid2D> (spectral)"] if self.is_spectral else ["<mikeio.Grid2D>"]
        )
        out.append(_print_axis_txt("x", self.x, self.dx))
        out.append(_print_axis_txt("y", self.y, self.dy))
        if self._is_rotated:
            ox, oy = self.origin
            out.append(
                f"origin: ({ox:.4g}, {oy:.4g}), orientation: {self._orientation:.3f}"
            )
        if self.projection_string:
            out.append(f"projection: {self.projection_string}")

        return "\n".join(out)

    def __str__(self) -> str:
        return f"Grid2D (ny={self.ny}, nx={self.nx})"

    @property
    def dx(self) -> float:
        """x grid spacing"""
        return self._dx

    @property
    def dy(self) -> float:
        """y grid spacing"""
        return self._dy

    @property
    def x(self) -> np.ndarray:
        """array of x coordinates (element center)"""
        if self.is_spectral and self.dx > 1:
            return self._logarithmic_f(self.nx, self._x0, self.dx)

        if self.is_local_coordinates and not (self.is_spectral or self.is_vertical):
            x0 = self._x0 + self._dx / 2
        else:
            x0 = self._x0

        x1 = x0 + self.dx * (self.nx - 1)
        x_local = np.linspace(x0, x1, self.nx)
        if self._is_rotated or self.is_spectral:
            return x_local
        else:
            return x_local + self._origin[0]

    @staticmethod
    def _logarithmic_f(
        n: int = 25, f0: float = 0.055, freq_factor: float = 1.1
    ) -> np.ndarray:
        """Generate logarithmic frequency axis

        Parameters
        ----------
        n : int, optional
            number of frequencies, by default 25
        f0 : float, optional
            Minimum frequency, by default 0.055
        freq_factor : float, optional
            Frequency factor, by default 1.1

        Returns
        -------
        np.ndarray
            array of logarithmic distributed discrete frequencies
        """
        logf0 = np.log(f0)
        logdf = np.log(f0 * freq_factor) - logf0
        logf = logf0 + logdf * np.arange(n)
        return np.exp(logf)

    @property
    def y(self) -> np.ndarray:
        """array of y coordinates (element center)"""
        if self.is_local_coordinates and not (self.is_spectral or self.is_vertical):
            y0 = self._y0 + self._dy / 2
        else:
            y0 = self._y0

        y1 = y0 + self.dy * (self.ny - 1)
        y_local = np.linspace(y0, y1, self.ny)
        return y_local if self._is_rotated else y_local + self._origin[1]

    @property
    def nx(self) -> int:
        """number of x grid points"""
        return self._nx

    @property
    def ny(self) -> int:
        """number of y grid points"""
        return self._ny

    @property
    def origin(self) -> Tuple[float, float]:
        """Coordinates of grid origo (in projection)"""
        return self._origin

    @property
    def orientation(self) -> float:
        """Grid orientation"""
        return self._orientation

    @property
    def bbox(self) -> BoundingBox:
        """bounding box (left, bottom, right, top)
        Note: not the same as the cell center values (x0,y0,x1,y1)!
        """
        if self._is_rotated:
            raise NotImplementedError("Only available if orientation = 0")
        if self.is_spectral:
            raise NotImplementedError("Not available for spectral Grid2D")
        left = self.x[0] - self.dx / 2
        bottom = self.y[0] - self.dy / 2
        right = self.x[-1] + self.dx / 2
        top = self.y[-1] + self.dy / 2
        return BoundingBox(left, bottom, right, top)

    @cached_property
    def xy(self) -> np.ndarray:
        """n-by-2 array of x- and y-coordinates"""
        xx, yy = np.meshgrid(self.x, self.y)
        xcol = xx.reshape(-1, 1)
        ycol = yy.reshape(-1, 1)
        return np.column_stack([xcol, ycol])

    @property
    def _cart(self) -> Cartography:
        """MIKE Core Cartography object"""
        factory = (
            Cartography.CreateGeoOrigin if self.is_geo else Cartography.CreateProjOrigin
        )
        return factory(self.projection_string, *self.origin, self.orientation)

    def _shift_x0y0_to_origin(self) -> None:
        """Shift spatial axis to start at (0,0) adding the start to origin instead
        Note: this will not change the x or y properties.
        """
        if self._is_rotated:
            raise ValueError("Only possible if orientation = 0")
            # TODO: needs testing
            # i0, j0 = self._x0/self.dx, self._y0/self.dy
            # self._x0, self._y0 = 0.0, 0.0
            # self._origin = self._cart.Xy2Proj(i0, j0)
        elif self.is_spectral:
            raise ValueError("Not possible for spectral Grid2D")
        else:
            x0, y0 = self._x0, self._y0
            self._x0, self._y0 = 0.0, 0.0
            self._origin = (self._origin[0] + x0, self._origin[1] + y0)

    def contains(self, coords: np.ndarray) -> Any:
        """test if a list of points are inside grid

        Parameters
        ----------
        coords : array(float)
            xy-coordinate of points given as n-by-2 array

        Returns
        -------
        bool array
            True for points inside, False otherwise
        """
        coords = np.atleast_2d(coords)
        y = coords[:, 1]
        x = coords[:, 0]

        xinside = (self.bbox.left <= x) & (x <= self.bbox.right)
        yinside = (self.bbox.bottom <= y) & (y <= self.bbox.top)
        return xinside & yinside

    def __contains__(self, pt: Any) -> Any:
        return self.contains(pt)

    def find_index(
        self,
        x: float | None = None,
        y: float | None = None,
        coords: np.ndarray | None = None,
        area: Tuple[float, float, float, float] | None = None,
    ) -> Tuple[Any, Any]:
        """Find nearest index (i,j) of point(s)

        Parameters
        ----------
        x : float
            x-coordinate of point
        y : float
            y-coordinate of point
        coords : array(float)
            xy-coordinate of points given as n-by-2 array
        area : array(float)
            xy-coordinates of bounding box

        Returns
        -------
        array(int), array(int)
            i- and j-index of nearest cell
        """
        if x is None and y is not None and not np.isscalar(y):
            raise ValueError(
                f"{y=} is not a scalar value, use the coords argument instead"
            )

        if y is None and x is not None and not np.isscalar(x):
            raise ValueError(
                f"{x=} is not a scalar value, use the coords argument instead"
            )

        if x is not None and y is not None:
            if coords is not None:
                raise ValueError("x,y and coords cannot be given at the same time!")
            coords = np.column_stack([np.atleast_1d(x), np.atleast_1d(y)])
        elif x is not None:
            if x < self.x[0] or x > self.x[-1]:
                raise OutsideModelDomainError(x=x, y=None)
            return np.atleast_1d(np.argmin(np.abs(self.x - x))), None
        elif y is not None:
            if y < self.y[0] or y > self.y[-1]:
                raise OutsideModelDomainError(x=None, y=y)
            return None, np.atleast_1d(np.argmin(np.abs(self.y - y)))

        if coords is not None:
            return self._xy_to_index(coords)
        elif area is not None:
            return self._bbox_to_index(area)
        else:
            raise ValueError("Provide x,y or coords")

    def _xy_to_index(self, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find specific points in this geometry"""
        xy = np.atleast_2d(xy)
        y = xy[:, 1]
        x = xy[:, 0]

        ii = (-999999999) * np.ones_like(x, dtype=int)
        jj = (-999999999) * np.ones_like(y, dtype=int)

        inside = self.contains(xy)
        if np.any(~inside):
            raise OutsideModelDomainError(x=x[~inside], y=y[~inside])

        # get index in x, and y for points inside based on the grid spacing and origin
        ii = np.floor((x - (self.x[0] - self.dx / 2)) / self.dx).astype(int)
        jj = np.floor((y - (self.y[0] - self.dy / 2)) / self.dy).astype(int)

        return ii, jj

    def _bbox_to_index(
        self, bbox: Tuple[float, float, float, float] | BoundingBox
    ) -> Tuple[range, range]:
        """Find subarea within this geometry"""
        if not (len(bbox) == 4):
            raise ValueError(
                "area most be a bounding box of coordinates e.g. area=(-10.0, 10.0 20.0, 30.0)"
            )

        x0, y0, x1, y1 = bbox
        if x0 > self.x[-1] or y0 > self.y[-1] or x1 < self.x[0] or y1 < self.y[0]:
            raise ValueError("area is outside grid")

        mask = (self.x >= x0) & (self.x <= x1)
        ii = np.where(mask)[0]
        mask = (self.y >= y0) & (self.y <= y1)
        jj = np.where(mask)[0]

        i = range(ii[0], ii[-1] + 1)
        j = range(jj[0], jj[-1] + 1)

        return i, j

    @overload
    def isel(self, idx: int, axis: int) -> "Grid1D": ...

    @overload
    def isel(self, idx: slice, axis: int) -> "Grid2D": ...

    def isel(
        self, idx: np.ndarray | int | slice, axis: int
    ) -> "Grid2D | Grid1D | GeometryUndefined":
        """Return a new geometry as a subset of Grid2D along the given axis."""
        assert isinstance(axis, int), "axis must be an integer (or 'x' or 'y')"
        axis = axis + 2 if axis < 0 else axis

        if not np.isscalar(idx):
            d = np.diff(idx)  # type: ignore
            if np.any(d < 1) or not np.allclose(d, d[0]):
                # non-equidistant grid is not supported by Dfs2
                return GeometryUndefined()
            else:
                ii = idx if axis == 1 else None
                jj = idx if axis == 0 else None
                return self._index_to_Grid2D(ii, jj)  # type: ignore

        if axis == 0:
            # y is first axis! if we select an element from y-axis (axis 0),
            # we return a "copy" of the x-axis
            nc = np.column_stack([self.x, self.y[idx] * np.ones_like(self.x)])  # type: ignore
            return Grid1D(x=self.x, projection=self.projection, node_coordinates=nc)
        elif axis == 1:
            nc = np.column_stack([self.x[idx] * np.ones_like(self.y), self.y])  # type: ignore
            return Grid1D(
                x=self.y, projection=self.projection, node_coordinates=nc, axis_name="y"
            )
        else:
            raise ValueError(f"axis must be 0 or 1 (or 'x' or 'y'), not {axis}")

    def _index_to_Grid2D(
        self, ii: range | None = None, jj: range | None = None
    ) -> "Grid2D | GeometryUndefined":
        ii = range(self.nx) if ii is None else ii
        jj = range(self.ny) if jj is None else jj
        assert len(ii) > 1 and len(jj) > 1, "Index must be at least len 2"
        assert ii[-1] < self.nx and jj[-1] < self.ny, "Index out of bounds"
        di = np.diff(ii)
        dj = np.diff(jj)
        if (np.any(di < 1) or not np.allclose(di, di[0])) or (
            np.any(dj < 1) or not np.allclose(dj, dj[0])
        ):
            # warnings.warn("Axis not equidistant! Will return GeometryUndefined()")
            raise ValueError()
            # return GeometryUndefined()
        else:
            dx = self.dx * di[0]
            dy = self.dy * dj[0]
            x0 = self._x0 + (self.x[ii[0]] - self.x[0])
            y0 = self._y0 + (self.y[jj[0]] - self.y[0])
            origin = None if self._shift_origin_on_write else self.origin
            # if not self._is_rotated and not self._shift_origin_on_write:
            if self._is_rotated:
                origin = self._cart.Xy2Proj(ii[0], jj[0])
                # what about the orientation if is_geo??
                # orientationGeo = proj.Proj2GeoRotation(east, north, orientationProj)
                x0, y0 = (0.0, 0.0)
            elif not self.is_spectral:
                origin = (self.origin[0] + x0, self.origin[1] + y0)
                x0, y0 = (0.0, 0.0)

            return Grid2D(
                x0=x0,
                y0=y0,
                dx=dx,
                dy=dy,
                nx=len(ii),
                ny=len(jj),
                projection=self.projection,
                orientation=self._orientation,
                is_spectral=self.is_spectral,
                is_vertical=self.is_vertical,
                origin=origin,
            )

    def _to_element_table(self, index_base: int = 0) -> List[List[int]]:
        elem_table = []
        for elx in range(self.nx - 1):
            # each col
            for ely in range(self.ny - 1):
                # each row (in this col)
                n1 = ely * self.nx + elx + index_base
                n2 = (ely + 1) * self.nx + elx + index_base
                elem_table.append([n1, n1 + 1, n2 + 1, n2])
        return elem_table

    @staticmethod
    def _centers_to_nodes(x: np.ndarray) -> np.ndarray:
        """Nodes will be placed mid-way between centers
        If non-equidistant, new centers will hence not equal old centers!
        """
        if len(x) == 1:
            return np.array([x[0] - 0.5, x[0] + 0.5])
        xinner = (x[1:] + x[:-1]) / 2
        left = x[0] - (x[1] - x[0]) / 2
        right = x[-1] + (x[-1] - x[-2]) / 2
        return np.array([left, *xinner, right])

    @staticmethod
    def _nodes_to_centers(xn: np.ndarray) -> Any:
        return (xn[1:] + xn[:-1]) / 2

    def get_node_coordinates(self) -> np.ndarray:
        """node coordinates for this grid

        Returns
        -------
        array(float)
            2d array with x,y-coordinates, length=(nx+1)*(ny+1)
        """
        xn = self._centers_to_nodes(self.x)
        yn = self._centers_to_nodes(self.y)
        gn = Grid2D(x=xn, y=yn)
        return gn.xy

    def to_geometryFM(
        self,
        *,
        z: float | None = None,
        west: int = 2,
        east: int = 4,
        north: int = 5,
        south: int = 3,
    ) -> GeometryFM2D:
        """convert Grid2D to GeometryFM2D

        Parameters
        ----------
        z : float, optional
            bathymetry values for each node, by default 0
        west: int, optional
            code value for west boundary
        east: int, optional
            code value for east boundary
        north: int, optional
            code value for north boundary
        south: int, optional
            code value for south boundary
        """
        from mikeio.spatial._FM_geometry import GeometryFM2D

        # get node based grid
        xn = self._centers_to_nodes(self.x)
        yn = self._centers_to_nodes(self.y)
        gn = Grid2D(x=xn, y=yn)

        # node coordinates
        x = gn.xy[:, 0]
        y = gn.xy[:, 1]
        n = gn.nx * gn.ny

        zn = np.zeros_like(x)
        if z is not None:
            zn[:] = z

        codes = np.zeros(n, dtype=int)
        codes[y == y[-1]] = north
        codes[x == x[-1]] = east
        codes[y == y[0]] = south
        codes[x == x[0]] = west
        codes[(y == y[-1]) & (x == x[0])] = 5  # corner->north

        nc = np.column_stack([x, y, zn])
        elem_table = gn._to_element_table(index_base=0)
        return GeometryFM2D(
            node_coordinates=nc,
            element_table=elem_table,
            codes=codes,
            projection=self.projection,
        )

    def to_mesh(
        self, outfilename: str | Path, z: np.ndarray | float | None = None
    ) -> None:
        """export grid to mesh file

        Parameters
        ----------
        outfilename : str
            path of new mesh file
        z : float or array(float), optional
            bathymetry values for each node, by default 0
            if array: must have length=(nx+1)*(ny+1)
        """
        g = self.to_geometryFM()

        if z is not None:
            if not np.isscalar(z):
                assert isinstance(z, np.ndarray), "z must be a numpy array"
                if len(z) != g.n_nodes:
                    raise ValueError(
                        "z must either be scalar or have length of nodes ((nx+1)*(ny+1))"
                    )
            g.node_coordinates[:, 2] = z
        g.to_mesh(outfilename=outfilename)


@dataclass
class Grid3D(_Geometry):
    """3D  grid
    Origin in the center of cell in lower-left corner
    x, y and z axes are increasing and equidistant
    """

    _dx: float
    _nx: int
    _x0: float
    _dy: float
    _ny: int
    _y0: float
    _dz: float
    _nz: int
    _z0: float
    _projstr: str
    _origin: Tuple[float, float]
    _orientation: float

    def __init__(
        self,
        *,
        x: np.ndarray | None = None,
        x0: float = 0.0,
        dx: float | None = None,
        nx: int | None = None,
        y: np.ndarray | None = None,
        y0: float = 0.0,
        dy: float | None = None,
        ny: int | None = None,
        z: np.ndarray | None = None,
        z0: float = 0.0,
        dz: float | None = None,
        nz: int | None = None,
        projection: str = "NON-UTM",  # TODO LONG/LAT
        origin: Tuple[float, float] = (0.0, 0.0),
        orientation: float = 0.0,
    ) -> None:
        super().__init__(projection=projection)
        self._origin = (0.0, 0.0) if origin is None else (origin[0], origin[1])
        assert len(self._origin) == 2, "origin must be a tuple of length 2"
        self._x0, self._dx, self._nx = _parse_grid_axis("x", x, x0, dx, nx)
        self._y0, self._dy, self._ny = _parse_grid_axis("y", y, y0, dy, ny)
        self._z0, self._dz, self._nz = _parse_grid_axis("z", z, z0, dz, nz)

        self._projstr = projection  # TODO handle other types than string
        self._origin = origin
        self._orientation = orientation

    @property
    def default_dims(self) -> Tuple[str, ...]:
        return ("z", "y", "x")

    @property
    def ndim(self) -> int:
        return 3

    @property
    def _is_rotated(self) -> Any:
        return np.abs(self._orientation) > 1e-5

    @property
    def x(self) -> np.ndarray:
        """array of x-axis coordinates (element center)"""
        x0 = self._x0 + self._dx / 2 if self.is_local_coordinates else self._x0

        x1 = x0 + self.dx * (self.nx - 1)
        x_local = np.linspace(x0, x1, self.nx)
        return x_local if self._is_rotated else x_local + self.origin[0]

    @property
    def dx(self) -> float:
        """x-axis grid spacing"""
        return self._dx

    @property
    def nx(self) -> int:
        """number of x grid points"""
        return self._nx

    @property
    def y(self) -> np.ndarray:
        """array of y-axis coordinates (element center)"""
        y0 = self._y0 + self._dy / 2 if self.is_local_coordinates else self._y0
        y1 = y0 + self.dy * (self.ny - 1)
        y_local = np.linspace(y0, y1, self.ny)
        return y_local if self._is_rotated else y_local + self.origin[1]

    @property
    def dy(self) -> float:
        """y-axis grid spacing"""
        return self._dy

    @property
    def ny(self) -> int:
        """number of y grid points"""
        return self._ny

    @property
    def z(self) -> np.ndarray:
        """array of z-axis node coordinates"""
        z1 = self._z0 + self.dz * (self.nz - 1)
        return np.linspace(self._z0, z1, self.nz)

    @property
    def dz(self) -> float:
        """z-axis grid spacing"""
        return self._dz

    @property
    def nz(self) -> int:
        """number of z grid points"""
        return self._nz

    @property
    def origin(self) -> Tuple[float, float]:
        """Coordinates of grid origo (in projection)"""
        return self._origin

    @property
    def orientation(self) -> float:
        """Grid orientation"""
        return self._orientation

    def find_index(
        self, coords: Any = None, layers: Any = None, area: Any = None
    ) -> Any:
        if layers is not None:
            raise NotImplementedError(
                f"Layer slicing is not yet implemented. Use the mikeio.read('file.dfs3', layers='{layers}')"
            )
        raise NotImplementedError(
            "Not yet implemented for Grid3D. Please use mikeio.read('file.dfs3') and its arguments instead."
        )

    def isel(
        self, idx: int | np.ndarray, axis: int
    ) -> Grid3D | Grid2D | GeometryUndefined:
        """Get a subset geometry from this geometry"""
        assert isinstance(axis, int), "axis must be an integer (or 'x', 'y' or 'z')"
        axis = axis + 3 if axis < 0 else axis

        if not np.isscalar(idx):
            assert isinstance(idx, np.ndarray), "idx must be a numpy array"
            d = np.diff(idx)
            if np.any(d < 1) or not np.allclose(d, d[0]):
                return GeometryUndefined()
            else:
                ii = idx if axis == 2 else None
                jj = idx if axis == 1 else None
                kk = idx if axis == 0 else None
                return self._index_to_Grid3D(ii, jj, kk)

        if axis == 0:
            # z is the first axis! return x-y Grid2D
            # TODO: origin, how to pass self.z[idx]?
            return Grid2D(
                x0=self._x0,
                y0=self._y0,
                nx=self.nx,
                ny=self.ny,
                dx=self.dx,
                dy=self.dy,
                projection=self.projection,
                origin=self.origin,
                orientation=self.orientation,
            )
        elif axis == 1:
            # y is the second axis! return x-z Grid2D
            # TODO: origin, how to pass self.y[idx]?
            return Grid2D(
                x=self.x,
                y=self.z,
                # projection=self._projection,
            )
        elif axis == 2:
            # x is the last axis! return y-z Grid2D
            # TODO: origin, how to pass self.x[idx]?
            return Grid2D(
                x=self.y,
                y=self.z,
                # projection=self._projection,
            )
        else:
            raise ValueError(f"axis must be 0, 1 or 2 (or 'x', 'y' or 'z'), not {axis}")

    def _index_to_Grid3D(
        self,
        ii: np.ndarray | range | None = None,
        jj: np.ndarray | range | None = None,
        kk: np.ndarray | range | None = None,
    ) -> Grid3D | GeometryUndefined:
        ii = range(self.nx) if ii is None else ii
        jj = range(self.ny) if jj is None else jj
        kk = range(self.nz) if kk is None else kk
        assert (
            len(ii) > 1 and len(jj) > 1 and len(kk) > 1
        ), "Index must be at least len 2"
        assert (
            ii[-1] < self.nx and jj[-1] < self.ny and kk[-1] < self.nz
        ), "Index out of bounds"
        di = np.diff(ii)
        dj = np.diff(jj)
        dk = np.diff(kk)
        if (
            (np.any(di < 1) or not np.allclose(di, di[0]))
            or (np.any(dj < 1) or not np.allclose(dj, dj[0]))
            or (np.any(dk < 1) or not np.allclose(dk, dk[0]))
        ):
            warnings.warn("Axis not equidistant! Will return GeometryUndefined()")
            return GeometryUndefined()
        else:
            dx = self.dx * di[0]
            dy = self.dy * dj[0]
            dz = self.dz * dk[0]
            x0 = self._x0 + (self.x[ii[0]] - self.x[0])
            y0 = self._y0 + (self.y[jj[0]] - self.y[0])
            z0 = self._z0 + (self.z[kk[0]] - self.z[0])
            if self._is_rotated:
                # rotated => most be projected
                cart = Cartography.CreateProjOrigin(
                    self.projection, *self.origin, self.orientation
                )
                origin = cart.Xy2Proj(ii[0], jj[0])
            else:
                origin = (self.origin[0] + x0, self.origin[1] + y0)

            x0, y0 = (0.0, 0.0)

            return Grid3D(
                x0=x0,
                y0=y0,
                z0=z0,
                dx=dx,
                dy=dy,
                dz=dz,
                nx=len(ii),
                ny=len(jj),
                nz=len(kk),
                projection=self.projection,
                orientation=self.orientation,
                origin=origin,
            )

    def __repr__(self) -> str:
        out = ["<mikeio.Grid3D>"]
        out.append(_print_axis_txt("x", self.x, self.dx))
        out.append(_print_axis_txt("y", self.y, self.dy))
        out.append(_print_axis_txt("z", self.z, self.dz))
        out.append(
            f"origin: ({self._origin[0]:.4g}, {self._origin[1]:.4g}), orientation: {self._orientation:.3f}"
        )
        if self.projection_string:
            out.append(f"projection: {self.projection_string}")
        return "\n".join(out)

    def __str__(self) -> str:
        return f"Grid3D(nz={self.nz}, ny={self.ny}, nx={self.nx})"

    def _geometry_for_layers(
        self, layers: Sequence[int] | None, keepdims: bool = False
    ) -> Grid2D | "Grid3D" | "GeometryUndefined":
        if layers is None:
            return self

        g = self
        if len(layers) == 1 and not keepdims:
            geometry_2d = Grid2D(
                dx=g._dx,
                dy=g._dy,
                nx=g._nx,
                ny=g._ny,
                x0=g._x0,
                y0=g._y0,
                origin=g.origin,
                projection=g._projstr,
                orientation=g.orientation,
            )
            return geometry_2d

        d = np.diff(g.z[layers])
        if len(d) > 0:
            if np.any(d < 1) or not np.allclose(d, d[0]):
                warnings.warn("Extracting non-equidistant layers! Cannot use Grid3D.")
                return GeometryUndefined()

        geometry = Grid3D(
            x0=g._x0,
            y0=g._y0,
            nx=g._nx,
            dx=g._dx,
            ny=g._ny,
            dy=g._dy,
            z=g.z[layers],
            origin=g._origin,
            projection=g.projection,
            orientation=g.orientation,
        )
        return geometry
