from __future__ import annotations
import warnings
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from functools import cached_property
from collections.abc import (
    Iterable,
    Sized,
    Sequence,
    Mapping,
    MutableMapping,
)
from typing import (
    Any,
    Union,
    Literal,
    TYPE_CHECKING,
    overload,
    Callable,
)


import numpy as np
import pandas as pd
from mikecore.DfsuFile import DfsuFileType

from ..eum import EUMType, EUMUnit, ItemInfo
from ._data_utils import _get_time_idx_list, _n_selected_timesteps

if TYPE_CHECKING:
    from ._dataset import Dataset
    import xarray


from ..spatial import (
    Grid1D,
    Grid2D,
    Grid3D,
    GeometryPoint2D,
    GeometryPoint3D,
    GeometryUndefined,
    GeometryFM2D,
    GeometryFM3D,
    GeometryFMAreaSpectrum,
    GeometryFMLineSpectrum,
    GeometryFMPointSpectrum,
    GeometryFMVerticalColumn,
    GeometryFMVerticalProfile,
)

# We need this type to know if we should keep zn
from ..spatial._FM_geometry_layered import _GeometryFMLayered

from .._spectral import calc_m0_from_spectrum
from ._data_plot import (
    _DataArrayPlotter,
    _DataArrayPlotterFM,
    _DataArrayPlotterGrid1D,
    _DataArrayPlotterGrid2D,
    _DataArrayPlotterAreaSpectrum,
    _DataArrayPlotterFMVerticalColumn,
    _DataArrayPlotterFMVerticalProfile,
    _DataArrayPlotterPointSpectrum,
    _DataArrayPlotterLineSpectrum,
)

GeometryType = Union[
    GeometryUndefined,
    GeometryPoint2D,
    GeometryPoint3D,
    GeometryFM2D,
    GeometryFM3D,
    GeometryFMAreaSpectrum,
    GeometryFMLineSpectrum,
    GeometryFMPointSpectrum,
    GeometryFMVerticalColumn,
    GeometryFMVerticalProfile,
    Grid1D,
    Grid2D,
    Grid3D,
]

IndexType = Union[int, slice, Sequence[int], np.ndarray, None]


class _DataArraySpectrumToHm0:
    def __init__(self, da: "DataArray") -> None:
        self.da = da

    def __call__(self, tail: bool = True) -> "DataArray":
        # TODO: if action_density
        m0 = calc_m0_from_spectrum(
            self.da.to_numpy(),
            self.da.frequencies,
            self.da.directions,
            tail,
        )
        Hm0 = 4 * np.sqrt(m0)
        dims = tuple([d for d in self.da.dims if d not in ("frequency", "direction")])
        item = ItemInfo(EUMType.Significant_wave_height)
        g = self.da.geometry
        geometry: Any = GeometryUndefined()

        if isinstance(g, GeometryFMLineSpectrum):
            geometry = Grid1D(
                nx=g.n_nodes,
                dx=1.0,
                node_coordinates=g.node_coordinates,
                axis_name="node",
            )
        elif isinstance(g, GeometryFMAreaSpectrum):
            geometry = GeometryFM2D(
                node_coordinates=g.node_coordinates,
                codes=g.codes,
                node_ids=g.node_ids,
                projection=g.projection_string,
                element_table=g.element_table,
                element_ids=g.element_ids,
            )

        return DataArray(
            data=Hm0,
            time=self.da.time,
            item=item,
            dims=dims,
            geometry=geometry,
            dt=self.da._dt,
        )


class DataArray:
    """DataArray with data and metadata for a single item in a dfs file.

    The DataArray has these main properties:

    * time - a pandas.DatetimeIndex with the time instances of the data
    * geometry - a geometry object e.g. Grid2D or GeometryFM
    * values - a numpy array containing the data
    * item - an ItemInfo with name, type and unit

    Examples
    --------
    ```{python}
    import pandas as pd
    import mikeio

    da = mikeio.DataArray([0.0, 1.0],
        time=pd.date_range("2020-01-01", periods=2),
        item=mikeio.ItemInfo("Water level", mikeio.EUMType.Water_Level))
    da
    ```

    """

    deletevalue = 1.0e-35

    def __init__(
        self,
        data: np.ndarray,
        *,
        time: pd.DatetimeIndex | str | None = None,
        item: ItemInfo | None = None,
        geometry: GeometryType | None = None,
        zn: np.ndarray | None = None,
        dims: Sequence[str] | None = None,
        dt: float = 1.0,
    ) -> None:
        # TODO: add optional validation validate=True
        self._values = self._parse_data(data)
        self.time: pd.DatetimeIndex = self._parse_time(time)
        self._dt = dt

        geometry = GeometryUndefined() if geometry is None else geometry
        self.dims = self._parse_dims(dims, geometry)

        self._check_time_data_length(self.time)

        self.item = self._parse_item(item)
        self.geometry = self._parse_geometry(geometry, self.dims, self.shape)
        self._zn = self._parse_zn(zn, self.geometry, self.n_timesteps)
        self._set_spectral_attributes(geometry)
        self.plot = self._get_plotter_by_geometry()

    @staticmethod
    def _parse_data(data: Any) -> Any:  # np.ndarray | float:
        if not hasattr(data, "shape"):
            try:
                data = np.array(data, dtype=float)
            except ValueError:
                raise ValueError("Data must be convertible to a numpy array")
        return data

    def _parse_dims(
        self, dims: Sequence[str] | None, geometry: GeometryType
    ) -> tuple[str, ...]:
        if dims is None:
            return self._guess_dims(self.ndim, self.shape, self.n_timesteps, geometry)
        else:
            if self.ndim != len(dims):
                raise ValueError("Number of named dimensions does not equal data ndim")
            if ("time" in dims) and dims[0] != "time":
                raise ValueError("time must be first dimension if present!")
            if (self.n_timesteps > 1) and ("time" not in dims):
                raise ValueError(
                    f"time missing from named dimensions {dims}! (number of timesteps: {self.n_timesteps})"
                )
            return tuple(dims)

    @staticmethod
    def _guess_dims(
        ndim: int, shape: tuple[int, ...], n_timesteps: int, geometry: GeometryType
    ) -> tuple[str, ...]:
        # This is not very robust, but is probably a reasonable guess
        time_is_first = (n_timesteps > 1) or (shape[0] == 1 and n_timesteps == 1)
        dims = ["time"] if time_is_first else []
        ndim_no_time = ndim if (len(dims) == 0) else ndim - 1

        if isinstance(geometry, GeometryUndefined):
            DIMS_MAPPING: Mapping[int, Sequence[Any]] = {
                0: [],
                1: ["x"],
                2: ["y", "x"],
                3: ["z", "y", "x"],
            }
            spdims = DIMS_MAPPING[ndim_no_time]
        else:
            spdims = geometry.default_dims
        dims.extend(spdims)  # type: ignore
        return tuple(dims)

    def _check_time_data_length(self, time: Sized) -> None:
        if "time" in self.dims and len(time) != self._values.shape[0]:
            raise ValueError(
                f"Number of timesteps ({len(time)}) does not fit with data shape {self.values.shape}"
            )

    @staticmethod
    def _parse_item(item: ItemInfo | str | EUMType | None) -> ItemInfo:
        if isinstance(item, ItemInfo):
            return item

        if item is None:
            return ItemInfo("NoName")

        if isinstance(item, (str, EUMType, EUMUnit)):
            return ItemInfo(item)

        raise ValueError("item must be str, EUMType or EUMUnit")

    @staticmethod
    def _parse_geometry(
        geometry: Any, dims: tuple[str, ...], shape: tuple[int, ...]
    ) -> Any:
        if len(dims) > 1 and (
            geometry is None or isinstance(geometry, GeometryUndefined)
        ):
            if dims == ("time", "x"):
                return Grid1D(nx=shape[1], dx=1.0 / (shape[1] - 1))

            warnings.warn("Geometry is required for ndim >=1")

        axis = 1 if "time" in dims else 0
        # dims_no_time = tuple([d for d in dims if d != "time"])
        # shape_no_time = shape[1:] if ("time" in dims) else shape

        if len(dims) == 1 and dims[0] == "time":
            if geometry is not None:
                # assert geometry.ndim == 0
                return geometry
            else:
                return GeometryUndefined()

        if isinstance(geometry, GeometryFMPointSpectrum):
            pass
        elif isinstance(geometry, GeometryFM2D):
            if geometry.is_spectral:
                if geometry._type == DfsuFileType.DfsuSpectral1D:
                    assert (
                        shape[axis] == geometry.n_nodes
                    ), "data shape does not match number of nodes"
                elif geometry._type == DfsuFileType.DfsuSpectral2D:
                    assert (
                        shape[axis] == geometry.n_elements
                    ), "data shape does not match number of elements"
            else:
                assert (
                    shape[axis] == geometry.n_elements
                ), "data shape does not match number of elements"
        elif isinstance(geometry, Grid1D):
            assert (
                shape[axis] == geometry.nx
            ), "data shape does not match number of grid points"
        elif isinstance(geometry, Grid2D):
            assert shape[axis] == geometry.ny, "data shape does not match ny"
            assert shape[axis + 1] == geometry.nx, "data shape does not match nx"
        # elif isinstance(geometry, Grid3D): # TODO

        return geometry

    @staticmethod
    def _parse_zn(
        zn: np.ndarray | None, geometry: GeometryType, n_timesteps: int
    ) -> np.ndarray | None:
        if zn is not None:
            if isinstance(geometry, _GeometryFMLayered):
                # TODO: np.squeeze(zn) if n_timesteps=1 ?
                if (n_timesteps > 1) and (zn.shape[0] != n_timesteps):
                    raise ValueError(
                        f"zn has wrong shape ({zn.shape}). First dimension should be of size n_timesteps ({n_timesteps})"
                    )
                if zn.shape[-1] != geometry.n_nodes:
                    raise ValueError(
                        f"zn has wrong shape ({zn.shape}). Last dimension should be of size n_nodes ({geometry.n_nodes})"
                    )
            else:
                raise ValueError("zn can only be provided for layered dfsu data")
        return zn

    def _is_compatible(self, other: "DataArray", raise_error: bool = False) -> bool:
        """check if other DataArray has equivalent dimensions, time and geometry."""
        problems = []
        assert isinstance(other, DataArray)
        if self.shape != other.shape:
            problems.append("shape of data must be the same")
        if self.n_timesteps != other.n_timesteps:
            problems.append("Number of timesteps must be the same")
        if self.start_time != other.start_time:
            problems.append("start_time must be the same")
        if not isinstance(self.geometry, other.geometry.__class__):
            problems.append("The type of geometry must be the same")
        if hasattr(self.geometry, "__eq__"):
            if not (self.geometry == self.geometry):
                problems.append("The geometries must be the same")
        if self._zn is not None:
            # it can be expensive to check equality of zn
            # so we test only size, first and last element
            if (
                other._zn is None
                or self._zn.shape != other._zn.shape
                or self._zn.ravel()[0] != other._zn.ravel()[0]
                or self._zn.ravel()[-1] != other._zn.ravel()[-1]
            ):
                problems.append("zn must be the same")

        if self.dims != other.dims:
            problems.append("Dimension names (dims) must be the same")

        if raise_error and len(problems) > 0:
            raise ValueError(", ".join(problems))

        return len(problems) == 0

    def _get_plotter_by_geometry(self) -> Any:
        # TODO: this is explicit, but with consistent naming, we could create this mapping automatically
        PLOTTER_MAP: Any = {
            GeometryFMVerticalProfile: _DataArrayPlotterFMVerticalProfile,
            GeometryFMVerticalColumn: _DataArrayPlotterFMVerticalColumn,
            GeometryFMPointSpectrum: _DataArrayPlotterPointSpectrum,
            GeometryFMLineSpectrum: _DataArrayPlotterLineSpectrum,
            GeometryFMAreaSpectrum: _DataArrayPlotterAreaSpectrum,
            GeometryFM2D: _DataArrayPlotterFM,
            GeometryFM3D: _DataArrayPlotterFM,
            Grid1D: _DataArrayPlotterGrid1D,
            Grid2D: _DataArrayPlotterGrid2D,
        }

        plotter = PLOTTER_MAP.get(type(self.geometry), _DataArrayPlotter)
        return plotter(self)

    def _set_spectral_attributes(self, geometry: GeometryType) -> None:
        if hasattr(geometry, "frequencies") and hasattr(geometry, "directions"):
            assert isinstance(
                geometry,
                (
                    GeometryFMAreaSpectrum,
                    GeometryFMLineSpectrum,
                    GeometryFMPointSpectrum,
                ),
            )
            self.frequencies = geometry.frequencies
            self.n_frequencies = geometry.n_frequencies
            self.directions = geometry.directions
            self.n_directions = geometry.n_directions
            self.to_Hm0 = _DataArraySpectrumToHm0(self)

    # ============= Basic properties/methods ===========

    @property
    def name(self) -> str:
        """Name of this DataArray (=da.item.name)."""
        assert isinstance(self.item.name, str)
        return self.item.name

    @name.setter
    def name(self, value: str) -> None:
        self.item.name = value

    @property
    def type(self) -> EUMType:
        """EUMType."""
        return self.item.type

    @property
    def unit(self) -> EUMUnit:
        """EUMUnit."""
        return self.item._unit

    @unit.setter
    def unit(self, value: EUMUnit) -> None:
        self.item.unit = value

    @property
    def start_time(self) -> datetime:
        """First time instance (as datetime)."""
        return self.time[0].to_pydatetime()

    @property
    def end_time(self) -> datetime:
        """Last time instance (as datetime)."""
        # TODO: use pd.Timestamp instead
        return self.time[-1].to_pydatetime()

    @cached_property
    def is_equidistant(self) -> bool:
        """Is DataArray equidistant in time?"""
        if len(self.time) < 3:
            return True
        return len(self.time.to_series().diff().dropna().unique()) == 1

    @property
    def timestep(self) -> float:
        """Time step in seconds if equidistant (and at
        least two time instances); otherwise original time step is returned.
        """
        dt = self._dt
        if len(self.time) > 1 and self.is_equidistant:
            first: pd.Timestamp = self.time[0]
            second: pd.Timestamp = self.time[1]
            dt = (second - first).total_seconds()
        return dt

    @property
    def n_timesteps(self) -> int:
        """Number of time steps."""
        return len(self.time)

    @property
    def shape(self) -> Any:
        """Tuple of array dimensions."""
        return self.values.shape

    @property
    def ndim(self) -> int:
        """Number of array dimensions."""
        assert isinstance(self.values.ndim, int)
        return self.values.ndim

    @property
    def dtype(self) -> Any:
        """Data-type of the array elements."""
        return self.values.dtype

    @property
    def values(self) -> np.ndarray:
        """Values as a np.ndarray (equivalent to to_numpy())."""
        return self._values

    @values.setter
    def values(self, value: np.ndarray | float) -> None:
        if np.isscalar(self._values):
            if not np.isscalar(value):
                raise ValueError("Shape of new data is wrong (should be scalar)")
        elif value.shape != self._values.shape:  # type: ignore
            raise ValueError("Shape of new data is wrong")

        self._values = value  # type: ignore

    def to_numpy(self) -> np.ndarray:
        """Values as a np.ndarray (equivalent to values)."""
        return self._values

    @property
    def _has_time_axis(self) -> bool:
        return self.dims[0][0] == "t"

    def fillna(self, value: float = 0.0) -> "DataArray":
        """Fill NA/NaN value.

        Parameters
        ----------
        value: float, optional
            Value used to fill missing values. Default is 0.0.

        Examples
        --------
        ```{python}
        import numpy as np
        import mikeio

        da = mikeio.DataArray([np.nan, 1.0])
        da
        ```

        ```{python}
        da.fillna(0.0)
        ```

        """
        da = self.copy()
        x = da.values
        x[np.isnan(x)] = value
        return da

    def dropna(self) -> "DataArray":
        """Remove time steps where values are NaN."""
        if not self._has_time_axis:
            raise ValueError("Not available if no time axis!")

        x = self.to_numpy()

        # this seems overly complicated...
        axes = tuple(range(1, x.ndim))
        idx = list(np.where(~np.isnan(x).all(axis=axes))[0])
        return self.isel(idx, axis=0)

    def flipud(self) -> "DataArray":
        """Flip upside down (on first non-time axis)."""
        first_non_t_axis = 1 if self._has_time_axis else 0
        self.values = np.flip(self.values, axis=first_non_t_axis)
        return self

    def describe(self, percentiles=None, include=None, exclude=None) -> pd.DataFrame:  # type: ignore
        """Generate descriptive statistics by wrapping [](`pandas.DataFrame.describe`).

        Parameters
        ----------
        percentiles : list-like of numbers, optional
            The percentiles to include in the output. All should fall between 0 and 1.
        include : 'all', list-like of dtypes or None (default), optional
            A white list of data types to include in the result.
        exclude : list-like of dtypes or None (default), optional
            A black list of data types to omit from the result.


        Returns
        -------
        pd.DataFrame

        """
        data = {}
        data[self.name] = self.to_numpy().ravel()
        df = pd.DataFrame(data).describe(
            percentiles=percentiles, include=include, exclude=exclude
        )

        return df

    def copy(self) -> "DataArray":
        """Make copy of DataArray."""
        return deepcopy(self)

    def squeeze(self) -> "DataArray":
        """Remove axes of length 1.

        Returns
        -------
        DataArray

        """
        data = np.squeeze(self.values)

        dims = [d for s, d in zip(self.shape, self.dims) if s != 1]

        # TODO: should geometry stay the same?
        return DataArray(
            data=data,
            time=self.time,
            item=self.item,
            geometry=self.geometry,
            zn=self._zn,
            dims=tuple(dims),
            dt=self._dt,
        )

    # ============= Select/interp ===========
    def __getitem__(self, key: Any) -> "DataArray":
        da = self
        dims = self.dims
        key = self._getitem_parse_key(key)
        for j, k in enumerate(key):
            if isinstance(k, Iterable) or k != slice(None):
                if dims[j] == "time":
                    # getitem accepts fancy indexing only for time
                    k = self._get_time_idx_list(self.time, k)
                    if self._n_selected_timesteps(self.time, k) == 0:
                        raise IndexError("No timesteps found!")
                da = da.isel(k, axis=dims[j])
        return da

    def _getitem_parse_key(self, key: Any) -> Any:
        key = key if isinstance(key, tuple) else (key,)
        if len(key) > len(self.dims):
            raise IndexError(
                f"Key has more dimensions ({len(key)}) than DataArray ({len(self.dims)})!"
            )
        return key

    def __setitem__(self, key: Any, value: np.ndarray) -> None:
        if self._is_boolean_mask(key):
            mask = key if isinstance(key, np.ndarray) else key.values
            return self._set_by_boolean_mask(self._values, mask, value)
        self._values[key] = value

    def isel(
        self,
        idx: IndexType = None,
        *,
        time: IndexType = None,
        x: IndexType = None,
        y: IndexType = None,
        z: IndexType = None,
        element: IndexType = None,
        node: IndexType = None,
        layer: IndexType = None,
        frequency: IndexType = None,
        direction: IndexType = None,
        axis: int | str = 0,
    ) -> "DataArray":
        """Return a new DataArray whose data is given by
        integer indexing along the specified dimension(s).

        Note that the data will be a _view_ of the original data
        if possible (single index or slice), otherwise a copy (fancy indexing)
        following NumPy convention.

        The spatial parameters available depend on the dims
        (i.e. geometry) of the DataArray:

        * Grid1D: x
        * Grid2D: x, y
        * Grid3D: x, y, z
        * GeometryFM: element

        Parameters
        ----------
        idx: int, scalar or array_like
            Index, or indices, along the specified dimension(s)
        axis: (int, str, None), optional
            axis number or "time", by default 0
        time : int, optional
            time index,by default None
        x : int, optional
            x index, by default None
        y : int, optional
            y index, by default None
        z : int, optional
            z index, by default None
        layer: int, optional
            layer index, only used in dfsu 3d
        direction: int, optional
            direction index, only used in sprectra
        frequency: int, optional
            frequencey index, only used in spectra
        node: int, optional
            node index, only used in spectra
        element : int, optional
            Bounding box of coordinates (left lower and right upper)
            to be selected, by default None

        Returns
        -------
        DataArray
            new DataArray with selected data

        See Also
        --------
        dims : Get axis names
        sel : Select data using labels

        Examples
        --------
        ```{python}
        da = mikeio.read("../data/europe_wind_long_lat.dfs2")[0]
        da
        ```

        ```{python}
        da.isel(time=-1)
        ```

        ```{python}
        da.isel(x=slice(10,20), y=slice(40,60))
        ```

        ```{python}
        da = mikeio.read("../data/oresund_sigma_z.dfsu").Temperature
        da.isel(element=range(200))
        ```

        """
        if isinstance(self.geometry, Grid2D) and (x is not None and y is not None):
            return self.isel(x=x).isel(y=y)
        kwargs = {
            k: v
            for k, v in dict(
                time=time,
                x=x,
                y=y,
                z=z,
                element=element,
                node=node,
                layer=layer,
                frequency=frequency,
                direction=direction,
            ).items()
            if v is not None
        }
        for dim in kwargs:
            if dim in self.dims:
                axis = dim
                if idx is not None:
                    raise NotImplementedError(
                        "Selecting on multiple dimensions in the same call, not yet implemented"
                    )
                idx = kwargs[dim]
            else:
                raise ValueError(f"{dim} is not present in {self.dims}")

        axis = self._parse_axis(self.shape, self.dims, axis)

        idx_slice = None
        if isinstance(idx, slice):
            idx_slice = idx
            assert isinstance(axis, int)
            idx = list(range(*idx.indices(self.shape[axis])))
        if idx is None or (not np.isscalar(idx) and len(idx) == 0):  # type: ignore
            raise ValueError(
                "Empty index is not allowed"
            )  # TODO other option would be to have a NullDataArray

        idx = np.atleast_1d(idx)
        single_index = len(idx) == 1
        idx = idx[0] if single_index else idx

        if axis == 0 and self.dims[0] == "time":
            time = self.time[idx]
            geometry = self.geometry
            zn = None if self._zn is None else self._zn[idx]
        else:
            time = self.time
            geometry = GeometryUndefined()
            zn = None
            if hasattr(self.geometry, "isel"):
                assert isinstance(axis, int)
                spatial_axis = axis - 1 if self.dims[0] == "time" else axis
                geometry = self.geometry.isel(idx, axis=spatial_axis)

            # TODO this is ugly
            if isinstance(geometry, _GeometryFMLayered):
                node_ids, _ = self.geometry._get_nodes_and_table_for_elements(
                    idx, node_layers="all"
                )
                zn = self._zn[:, node_ids]  # type: ignore

        # reduce dims only if singleton idx
        dims = (
            tuple([d for i, d in enumerate(self.dims) if i != axis])
            if single_index
            else self.dims
        )
        if single_index:
            idx = int(idx)
        elif idx_slice is not None:
            idx = idx_slice

        if axis == 0:
            dat = self.values[idx]
        elif axis == 1:
            dat = self.values[:, idx]
        elif axis == 2:
            dat = self.values[:, :, idx]
        elif axis == 3:
            dat = self.values[:, :, :, idx]
        else:
            raise ValueError(f"Subsetting with {axis=} is not supported")

        return DataArray(
            data=dat,
            time=time,
            item=deepcopy(self.item),
            geometry=geometry,
            zn=zn,
            dims=dims,
            dt=self._dt,
        )

    def sel(
        self,
        *,
        time: str | pd.DatetimeIndex | "DataArray" | None = None,
        x: float | slice | None = None,
        y: float | slice | None = None,
        z: float | slice | None = None,
        coords: np.ndarray | None = None,
        area: tuple[float, float, float, float] | None = None,
        layers: int | str | Sequence[int | str] | None = None,
    ) -> "DataArray":
        """Return a new DataArray whose data is given by
        selecting index labels along the specified dimension(s).

        In contrast to DataArray.isel, indexers for this method
        should use labels instead of integers.

        The spatial parameters available depend on the geometry of the DataArray:

        * Grid1D: x
        * Grid2D: x, y, coords, area
        * Grid3D: [not yet implemented! use isel instead]
        * GeometryFM: (x,y), coords, area
        * GeometryFMLayered: (x,y,z), coords, area, layers

        Parameters
        ----------
        time : str, pd.DatetimeIndex, DataArray, optional
            time labels e.g. "2018-01" or slice("2018-1-1","2019-1-1"),
            by default None
        x : float, optional
            x-coordinate of point to be selected, by default None
        y : float, optional
            y-coordinate of point to be selected, by default None
        z : float, optional
            z-coordinate of point to be selected, by default None
        coords : np.array(float,float), optional
            As an alternative to specifying x, y and z individually,
            the argument coords can be used instead.
            (x,y)- or (x,y,z)-coordinates of point to be selected,
            by default None
        area : (float, float, float, float), optional
            Bounding box of coordinates (left lower and right upper)
            to be selected, by default None
        layers : int or str or list, optional
            layer(s) to be selected: "top", "bottom" or layer number
            from bottom 0,1,2,... or from the top -1,-2,... or as
            list of these; only for layered dfsu, by default None

        Returns
        -------
        DataArray
            new DataArray with selected data

        See Also
        --------
        isel : Select data using integer indexing
        interp : Interp data in time and space

        Examples
        --------
        ```{python}
        da = mikeio.read("../data/random.dfs1")[0]
        da
        ```
        ```{python}
        da.sel(time=slice(None, "2012-1-1 00:02"))
        ```

        ```{python}
        da.sel(x=100)
        ```
        ```{python}
        da = mikeio.read("../data/oresund_sigma_z.dfsu").Temperature
        da
        ```

        ```{python}
        da.sel(time="1997-09-15")
        ```

        ```{python}
        da.sel(x=340000, y=6160000, z=-3)
        ```
        ```{python}
        da.sel(layers="bottom")
        ```

        """
        # time is not part of kwargs
        kwargs = {
            k: v
            for k, v in dict(
                x=x, y=y, z=z, area=area, coords=coords, layers=layers
            ).items()
            if v is not None
        }
        if any([isinstance(v, slice) for v in kwargs.values()]):
            return self._sel_with_slice(kwargs)  # type: ignore

        da = self

        # select in space
        if len(kwargs) > 0:
            idx = self.geometry.find_index(**kwargs)

            # TODO this seems fragile
            if isinstance(idx, tuple):
                # TODO: support for dfs3
                assert len(idx) == 2
                ii, jj = idx
                if jj is not None:
                    da = da.isel(y=jj)
                if ii is not None:
                    da = da.isel(x=ii)
            else:
                da = da.isel(idx, axis="space")

        # select in time
        if time is not None:
            if hasattr(time, "time"):
                if isinstance(time.time, pd.DatetimeIndex):
                    time = time.time

            da = da[time]  # __getitem__ is 🚀

        return da

    def _sel_with_slice(self, kwargs: Mapping[str, slice]) -> "DataArray":
        for k, v in kwargs.items():
            if isinstance(v, slice):
                idx_start = (
                    self.geometry.find_index(**{k: v.start})
                    if v.start is not None
                    else None
                )
                idx_stop = (
                    self.geometry.find_index(**{k: v.stop})
                    if v.stop is not None
                    else None
                )
                pos = 0
                if isinstance(idx_start, tuple):
                    if k == "x":
                        pos = 0
                    if k == "y":
                        pos = 1

                start = idx_start[pos][0] if idx_start is not None else None
                stop = idx_stop[pos][0] if idx_stop is not None else None

                idx = slice(start, stop)

                self = self.isel(idx, axis=k)

        return self

    def interp(
        # TODO find out optimal syntax to allow interpolation to single point, new time, grid, mesh...
        self,
        # *, # TODO: make this a keyword-only argument in the future
        time: pd.DatetimeIndex | "DataArray" | None = None,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        n_nearest: int = 3,
        interpolant: tuple[Any, Any] | None = None,
        **kwargs: Any,
    ) -> "DataArray":
        """Interpolate data in time and space.

        This method currently has limited functionality for
        spatial interpolation. It will be extended in the future.

        The spatial parameters available depend on the geometry of the Dataset:

        * Grid1D: x
        * Grid2D: x, y
        * Grid3D: [not yet implemented!]
        * GeometryFM: (x,y)
        * GeometryFMLayered: (x,y) [surface point will be returned!]

        Parameters
        ----------
        time : float, pd.DatetimeIndex or DataArray, optional
            timestep in seconds or discrete time instances given by
            pd.DatetimeIndex (typically from another DataArray
            da2.time), by default None (=don't interp in time)
        x : float, optional
            x-coordinate of point to be interpolated to, by default None
        y : float, optional
            y-coordinate of point to be interpolated to, by default None
        z : float, optional
            z-coordinate of point to be interpolated to, by default None
        n_nearest : int, optional
            When using IDW interpolation, how many nearest points should
            be used, by default: 3
        interpolant : tuple, optional
            Precomputed interpolant, by default None
        **kwargs: Any
            Additional keyword arguments to be passed to the interpolation

        Returns
        -------
        DataArray
            new DataArray with interped data

        See Also
        --------
        sel : Select data using label indexing
        interp_like : Interp to another time/space of another DataArray
        interp_time : Interp in the time direction only

        Examples
        --------
        >>> da = mikeio.read("random.dfs1")[0]
        >>> da.interp(time=3600)
        >>> da.interp(x=110)

        >>> da = mikeio.read("HD2D.dfsu").Salinity
        >>> da.interp(x=340000, y=6160000)

        """
        if z is not None:
            raise NotImplementedError()

        geometry: GeometryPoint2D | GeometryPoint3D | GeometryUndefined = (
            GeometryUndefined()
        )

        # interp in space
        if (x is not None) or (y is not None) or (z is not None):
            coords = [(x, y)]

            if isinstance(self.geometry, Grid2D):  # TODO DIY bilinear interpolation
                if x is None or y is None:
                    raise ValueError("both x and y must be specified")

                xr_da = self.to_xarray()
                dai = xr_da.interp(x=x, y=y).values
                geometry = GeometryPoint2D(
                    x=x, y=y, projection=self.geometry.projection
                )
            elif isinstance(self.geometry, Grid1D):
                if interpolant is None:
                    interpolant = self.geometry.get_spatial_interpolant(coords)  # type: ignore
                dai = self.geometry.interp(self.to_numpy(), *interpolant).flatten()
                geometry = GeometryUndefined()
            elif isinstance(self.geometry, GeometryFM3D):
                raise NotImplementedError("Interpolation in 3d is not yet implemented")
            elif isinstance(self.geometry, GeometryFM2D):
                if x is None or y is None:
                    raise ValueError("both x and y must be specified")

                if interpolant is None:
                    interpolant = self.geometry.get_2d_interpolant(
                        coords,  # type: ignore
                        n_nearest=n_nearest,
                        **kwargs,  # type: ignore
                    )
                dai = self.geometry.interp2d(self, *interpolant).flatten()  # type: ignore
                if z is None:
                    geometry = GeometryPoint2D(
                        x=x, y=y, projection=self.geometry.projection
                    )
                # this is not supported yet (see above)
                # else:
                #    geometry = GeometryPoint3D(
                #        x=x, y=y, z=z, projection=self.geometry.projection
                #    )

            da = DataArray(
                data=dai,
                time=self.time,
                geometry=geometry,
                item=deepcopy(self.item),
                dt=self._dt,
            )
        else:
            da = self.copy()

        # interp in time
        if time is not None:
            da = da.interp_time(time)

        return da

    def __dataarray_read_item_time_func(
        self, item: int, step: int
    ) -> tuple[np.ndarray, float]:
        "Used by _extract_track."
        # Ignore item argument
        data = self.isel(time=step).to_numpy()
        time = (self.time[step] - self.time[0]).total_seconds()  # type: ignore

        return data, time

    def extract_track(
        self,
        track: pd.DataFrame,
        method: Literal["nearest", "inverse_distance"] = "nearest",
        dtype: Any = np.float32,
    ) -> "Dataset":
        """Extract data along a moving track.

        Parameters
        ---------
        track: pandas.DataFrame
            with DatetimeIndex and (x, y) of track points as first two columns
            x,y coordinates must be in same coordinate system as dfsu
        track: str
            filename of csv or dfs0 file containing t,x,y
        method: str, optional
            Spatial interpolation method ('nearest' or 'inverse_distance')
            default='nearest'
        dtype: Any, optional
            Data type of the output data, default=np.float32

        Returns
        -------
        Dataset
            A dataset with data dimension t
            The first two items will be x- and y- coordinates of track

        """
        from .._track import _extract_track

        assert self.timestep is not None

        return _extract_track(
            deletevalue=self.deletevalue,
            start_time=self.start_time,
            end_time=self.end_time,
            timestep=self.timestep,
            geometry=self.geometry,
            n_elements=self.shape[1],  # TODO is there a better way to find out this?
            track=track,
            items=deepcopy([self.item]),
            time_steps=list(range(self.n_timesteps)),
            item_numbers=[0],
            method=method,
            dtype=dtype,
            data_read_func=self.__dataarray_read_item_time_func,
        )

    def interp_time(
        self,
        dt: float | pd.DatetimeIndex | "DataArray",
        *,
        method: str = "linear",
        extrapolate: bool = True,
        fill_value: float = np.nan,
    ) -> "DataArray":
        """Temporal interpolation.

        Wrapper of [](`scipy.interpolate.interp1d`)

        Parameters
        ----------
        dt: float or pd.DatetimeIndex or Dataset/DataArray
            output timestep in seconds or new time axis
        method: str or int, optional
            Specifies the kind of interpolation as a string ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next', where 'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline interpolation of zeroth, first, second or third order; 'previous' and 'next' simply return the previous or next value of the point) or as an integer specifying the order of the spline interpolator to use. Default is 'linear'.
        extrapolate: bool, optional
            Default True. If False, a ValueError is raised any time interpolation is attempted on a value outside of the range of x (where extrapolation is necessary). If True, out of bounds values are assigned fill_value
        fill_value: float or array-like, optional
            Default NaN. this value will be used to fill in for points outside of the time range.

        Returns
        -------
        DataArray

        """
        from scipy.interpolate import interp1d  # type: ignore

        t_out_index = self._parse_interp_time(self.time, dt)
        t_in = self.time.values.astype(float)
        t_out = t_out_index.values.astype(float)

        data = interp1d(
            t_in,
            self.to_numpy(),
            axis=0,
            kind=method,
            bounds_error=not extrapolate,
            fill_value=fill_value,
        )(t_out)

        zn = (
            None
            if self._zn is None
            else interp1d(
                t_in,
                self._zn,
                axis=0,
                kind=method,
                bounds_error=not extrapolate,
                fill_value=fill_value,
            )(t_out)
        )

        return DataArray(
            data=data,
            time=t_out_index,
            item=deepcopy(self.item),
            geometry=self.geometry,
            zn=zn,
            dt=self._dt,
        )

    def interp_na(self, axis: str = "time", **kwargs: Any) -> "DataArray":
        """Fill in NaNs by interpolating according to different methods.

        Wrapper of [](`xarray.DataArray.interpolate_na`)

        Examples
        --------

        ```{python}
        import numpy as np
        import pandas as pd
        time = pd.date_range("2000", periods=3, freq="D")
        da = mikeio.DataArray(data=np.array([0.0, np.nan, 2.0]), time=time)
        da
        ```

        ```{python}
        da.interp_na()
        ```

        """
        xr_da = self.to_xarray().interpolate_na(dim=axis, **kwargs)
        self.values = xr_da.values
        return self

    def interp_like(
        self,
        other: "DataArray" | Grid2D | GeometryFM2D | pd.DatetimeIndex,
        interpolant: tuple[Any, Any] | None = None,
        **kwargs: Any,
    ) -> "DataArray":
        """Interpolate in space (and in time) to other geometry (and time axis).

        Note: currently only supports interpolation from dfsu-2d to
              dfs2 or other dfsu-2d DataArrays

        Parameters
        ----------
        other: Dataset, DataArray, Grid2D, GeometryFM, pd.DatetimeIndex
            The target geometry (and time axis) to interpolate to
        interpolant: tuple, optional
            Reuse pre-calculated index and weights
        **kwargs: Any
            additional kwargs are passed to interpolation method

        Examples
        --------
        >>> dai = da.interp_like(da2)
        >>> dae = da.interp_like(da2, extrapolate=True)
        >>> dat = da.interp_like(da2.time)

        Returns
        -------
        DataArray
            Interpolated DataArray

        """
        if not (isinstance(self.geometry, GeometryFM2D) and self.geometry.is_2d):
            raise NotImplementedError(
                "Currently only supports interpolating from 2d flexible mesh data!"
            )

        if isinstance(other, pd.DatetimeIndex):
            return self.interp_time(other, **kwargs)

        if not (isinstance(self.geometry, GeometryFM2D) and self.geometry.is_2d):
            raise NotImplementedError("Currently only supports 2d flexible mesh data!")

        if hasattr(other, "geometry"):
            geom = other.geometry
        else:
            geom = other

        if isinstance(geom, Grid2D):
            xy = geom.xy
        elif isinstance(geom, GeometryFM2D):
            xy = geom.element_coordinates[:, :2]
            if geom.is_layered:
                raise NotImplementedError(
                    "Does not yet support layered flexible mesh data!"
                )
        else:
            raise NotImplementedError()

        if interpolant is None:
            elem_ids, weights = self.geometry.get_2d_interpolant(xy, **kwargs)
        else:
            elem_ids, weights = interpolant

        if isinstance(geom, (Grid2D, GeometryFM2D)):
            shape = (geom.ny, geom.nx) if isinstance(geom, Grid2D) else None

            ari = self.geometry.interp2d(
                data=self.to_numpy(), elem_ids=elem_ids, weights=weights, shape=shape
            )
        else:
            raise NotImplementedError(
                "Interpolation to other geometry not yet supported"
            )
        assert isinstance(ari, np.ndarray)
        dai = DataArray(
            data=ari,
            time=self.time,
            geometry=geom,
            item=deepcopy(self.item),
            dt=self._dt,
        )

        if hasattr(other, "time"):
            dai = dai.interp_time(other.time)

        assert isinstance(dai, DataArray)

        return dai

    @staticmethod
    def concat(
        dataarrays: Sequence["DataArray"], keep: Literal["last", "first"] = "last"
    ) -> "DataArray":
        """Concatenate DataArrays along the time axis.

        Parameters
        ---------
        dataarrays: list[DataArray]
            DataArrays to concatenate
        keep: 'first' or 'last', optional
            default: last

        Returns
        -------
        DataArray
            The concatenated DataArray

        Examples
        --------
        ```{python}
        da1 = mikeio.read("../data/HD2D.dfsu", time=[0,1])[0]
        da2 = mikeio.read("../data/HD2D.dfsu", time=[2,3])[0]
        da1.time
        ```

        ```{python}
        da3 = mikeio.DataArray.concat([da1,da2])
        da3
        ```

        """
        from mikeio import Dataset

        datasets = [Dataset([da]) for da in dataarrays]

        ds = Dataset.concat(datasets, keep=keep)
        da = ds[0]
        assert isinstance(da, DataArray)
        return da

    # ============= Aggregation methods ===========

    def max(self, axis: int | str = 0, **kwargs: Any) -> "DataArray":
        """Max value along an axis.

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default 0
        **kwargs: Any
            Additional keyword arguments

        Returns
        -------
        DataArray
            array with max values

        See Also
        --------
            nanmax : Max values with NaN values removed

        """
        return self.aggregate(axis=axis, func=np.max, **kwargs)

    def min(self, axis: int | str = 0, **kwargs: Any) -> "DataArray":
        """Min value along an axis.

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default 0
        **kwargs: Any
            Additional keyword arguments

        Returns
        -------
        DataArray
            array with min values

        See Also
        --------
            nanmin : Min values with NaN values removed

        """
        return self.aggregate(axis=axis, func=np.min, **kwargs)

    def mean(self, axis: int | str = 0, **kwargs: Any) -> "DataArray":
        """Mean value along an axis.

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default 0
        **kwargs: Any
            Additional keyword arguments

        Returns
        -------
        DataArray
            array with mean values

        See Also
        --------
            nanmean : Mean values with NaN values removed

        """
        return self.aggregate(axis=axis, func=np.mean, **kwargs)

    def std(self, axis: int | str = 0, **kwargs: Any) -> "DataArray":
        """Standard deviation values along an axis.

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default 0
        **kwargs: Any
            Additional keyword arguments

        Returns
        -------
        DataArray
            array with standard deviation values

        See Also
        --------
            nanstd : Standard deviation values with NaN values removed

        """
        return self.aggregate(axis=axis, func=np.std, **kwargs)

    def ptp(self, axis: int | str = 0, **kwargs: Any) -> "DataArray":
        """Range (max - min) a.k.a Peak to Peak along an axis.

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default 0
        **kwargs: Any
            Additional keyword arguments

        Returns
        -------
        DataArray
            array with peak to peak values

        """
        return self.aggregate(axis=axis, func=np.ptp, **kwargs)

    def average(
        self, weights: np.ndarray, axis: int | str = 0, **kwargs: Any
    ) -> "DataArray":
        """Compute the weighted average along the specified axis.

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default
        weights: np.ndarray
            weights to apply to the values
        **kwargs: Any
            Additional keyword arguments

        Returns
        -------
        DataArray
            DataArray with weighted average values

        See Also
        --------
            aggregate : Weighted average

        Examples
        --------
        >>> dfs = Dfsu("HD2D.dfsu")
        >>> da = dfs.read(["Current speed"])[0]
        >>> area = dfs.get_element_area()
        >>> da2 = da.average(axis="space", weights=area)

        """

        def func(x, axis, keepdims):  # type: ignore
            if keepdims:
                raise NotImplementedError()

            return np.average(x, weights=weights, axis=axis)

        return self.aggregate(axis=axis, func=func, **kwargs)

    def nanmax(self, axis: int | str = 0, **kwargs: Any) -> "DataArray":
        """Max value along an axis (NaN removed).

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default 0
        **kwargs: Any
            Additional keyword arguments

        Returns
        -------
        DataArray
            array with max values

        See Also
        --------
            nanmax : Max values with NaN values removed

        """
        return self.aggregate(axis=axis, func=np.nanmax, **kwargs)

    def nanmin(self, axis: int | str = 0, **kwargs: Any) -> "DataArray":
        """Min value along an axis (NaN removed).

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default 0
        **kwargs: Any
            Additional keyword arguments

        Returns
        -------
        DataArray
            array with min values

        See Also
        --------
            nanmin : Min values with NaN values removed

        """
        return self.aggregate(axis=axis, func=np.nanmin, **kwargs)

    def nanmean(self, axis: int | str = 0, **kwargs: Any) -> "DataArray":
        """Mean value along an axis (NaN removed).

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default 0
        **kwargs: Any
            Additional keyword arguments

        Returns
        -------
        DataArray
            array with mean values

        See Also
        --------
            mean : Mean values

        """
        return self.aggregate(axis=axis, func=np.nanmean, **kwargs)

    def nanstd(self, axis: int | str = 0, **kwargs: Any) -> "DataArray":
        """Standard deviation value along an axis (NaN removed).

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default 0
        **kwargs: Any
            Additional keyword arguments

        Returns
        -------
        DataArray
            array with standard deviation values

        See Also
        --------
            std : Standard deviation

        """
        return self.aggregate(axis=axis, func=np.nanstd, **kwargs)

    def aggregate(
        self, axis: int | str = 0, func: Callable[..., Any] = np.nanmean, **kwargs: Any
    ) -> "DataArray":
        """Aggregate along an axis.

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default 0
        func: function, optional
            default np.nanmean
        **kwargs: Any
            Additional keyword arguments

        Returns
        -------
        DataArray
            dataarray with aggregated values

        See Also
        --------
            max : Max values
            nanmax : Max values with NaN values removed

        """
        axis = self._parse_axis(self.shape, self.dims, axis)
        time = self._time_by_agg_axis(self.time, axis)

        if isinstance(axis, int):
            axes = (axis,)
        else:
            axes = axis  # type: ignore

        dims = tuple([d for i, d in enumerate(self.dims) if i not in axes])

        item = deepcopy(self.item)
        if "name" in kwargs:
            item.name = kwargs.pop("name")

        with (
            warnings.catch_warnings()
        ):  # there might be all-Nan slices, it is ok, so we ignore them!
            warnings.simplefilter("ignore", category=RuntimeWarning)
            data = func(self.to_numpy(), axis=axis, keepdims=False, **kwargs)

        if axis == 0 and "time" in self.dims:  # time
            geometry = self.geometry
            zn = None if self._zn is None else self._zn[0]

        else:
            geometry = GeometryUndefined()
            zn = None

        return DataArray(
            data=data,
            time=time,
            item=item,
            geometry=geometry,
            dims=dims,
            zn=zn,
            dt=self._dt,
        )

    @overload
    def quantile(self, q: float, **kwargs: Any) -> "DataArray": ...

    @overload
    def quantile(self, q: Sequence[float], **kwargs: Any) -> "Dataset": ...

    def quantile(
        self, q: float | Sequence[float], *, axis: int | str = 0, **kwargs: Any
    ) -> "DataArray" | "Dataset":
        """Compute the q-th quantile of the data along the specified axis.

        Wrapping np.quantile

        Parameters
        ----------
        q: array_like of float
            Quantile or sequence of quantiles to compute,
            which must be between 0 and 1 inclusive.
        axis: (int, str, None), optional
            axis number or "time" or "space", by default 0
        **kwargs: Any
            Additional keyword arguments

        Returns
        -------
        DataArray
            data with quantile values

        Examples
        --------
        >>> da.quantile(q=[0.25,0.75])
        >>> da.quantile(q=0.5)
        >>> da.quantile(q=[0.01,0.5,0.99], axis="space")

        See Also
        --------
        nanquantile : quantile with NaN values ignored

        """
        return self._quantile(q, axis=axis, func=np.quantile, **kwargs)

    @overload
    def nanquantile(self, q: float, **kwargs: Any) -> "DataArray": ...

    @overload
    def nanquantile(self, q: Sequence[float], **kwargs: Any) -> "Dataset": ...

    def nanquantile(
        self, q: float | Sequence[float], *, axis: int | str = 0, **kwargs: Any
    ) -> "DataArray" | "Dataset":
        """Compute the q-th quantile of the data along the specified axis, while ignoring nan values.

        Wrapping np.nanquantile

        Parameters
        ----------
        q: array_like of float
            Quantile or sequence of quantiles to compute,
            which must be between 0 and 1 inclusive.
        axis: (int, str, None), optional
            axis number or "time" or "space", by default 0
        **kwargs: Any
            Additional keyword arguments

        Returns
        -------
        DataArray
            data with quantile values

        Examples
        --------
        >>> da.nanquantile(q=[0.25,0.75])
        >>> da.nanquantile(q=0.5)
        >>> da.nanquantile(q=[0.01,0.5,0.99], axis="space")

        See Also
        --------
        quantile : Quantile with NaN values

        """
        return self._quantile(q, axis=axis, func=np.nanquantile, **kwargs)

    def _quantile(self, q, *, axis: int | str = 0, func=np.quantile, **kwargs: Any):  # type: ignore
        from mikeio import Dataset

        axis = self._parse_axis(self.shape, self.dims, axis)
        assert isinstance(axis, int)
        time = self._time_by_agg_axis(self.time, axis)

        if np.isscalar(q):
            qdat = func(self.values, q=q, axis=axis, **kwargs)
            geometry = self.geometry if axis == 0 else GeometryUndefined()
            zn = self._zn if axis == 0 else None

            dims = tuple([d for i, d in enumerate(self.dims) if i != axis])
            item = deepcopy(self.item)
            return DataArray(
                data=qdat,
                time=time,
                item=item,
                geometry=geometry,
                dims=dims,
                zn=zn,
                dt=self._dt,
            )
        else:
            res = []
            for quantile in q:
                qd = self._quantile(q=quantile, axis=axis, func=func)
                newname = f"Quantile {quantile}, {self.name}"
                qd.name = newname
                res.append(qd)

            return Dataset(data=res, validate=False)

    # ============= MATH operations ===========

    def __radd__(self, other: "DataArray" | float) -> "DataArray":
        return self.__add__(other)

    def __add__(self, other: "DataArray" | float) -> "DataArray":
        return self._apply_math_operation(other, np.add)

    def __rsub__(self, other: "DataArray" | float) -> "DataArray":
        return other + self.__neg__()

    def __sub__(self, other: "DataArray" | float) -> "DataArray":
        return self._apply_math_operation(other, np.subtract)

    def __rmul__(self, other: "DataArray" | float) -> "DataArray":
        return self.__mul__(other)

    def __mul__(self, other: "DataArray" | float) -> "DataArray":
        return self._apply_math_operation(other, np.multiply)

    def __pow__(self, other: float) -> "DataArray":
        return self._apply_math_operation(other, np.power)

    def __truediv__(self, other: "DataArray" | float) -> "DataArray":
        return self._apply_math_operation(other, np.divide)

    def __floordiv__(self, other: "DataArray" | float) -> "DataArray":
        return self._apply_math_operation(other, np.floor_divide)

    def __mod__(self, other: float) -> "DataArray":
        return self._apply_math_operation(other, np.mod)

    def __neg__(self) -> "DataArray":
        return self._apply_unary_math_operation(np.negative)

    def __pos__(self) -> "DataArray":
        return self._apply_unary_math_operation(np.positive)

    def __abs__(self) -> "DataArray":
        return self._apply_unary_math_operation(np.abs)

    def _apply_unary_math_operation(self, func: Callable) -> "DataArray":
        try:
            data = func(self.values)

        except TypeError:
            raise TypeError("Math operation could not be applied to DataArray")

        new_da = self.copy()
        new_da.values = data
        return new_da

    def _apply_math_operation(
        self,
        other: "DataArray" | float,
        func: Callable,
    ) -> "DataArray":
        """Apply a binary math operation with a scalar, an array or another DataArray."""
        try:
            other_values = other.values if hasattr(other, "values") else other
            data = func(self.values, other_values)
        except TypeError:
            raise TypeError("Math operation could not be applied to DataArray")

        new_da = self.copy()  # TODO: alternatively: create new dataset (will validate)
        new_da.values = data

        return new_da

    # ============= Logical indexing ===========

    def __lt__(self, other) -> "DataArray":  # type: ignore
        bmask = self.values < self._other_to_values(other)
        return self._boolmask_to_new_DataArray(bmask)

    def __gt__(self, other) -> "DataArray":  # type: ignore
        bmask = self.values > self._other_to_values(other)
        return self._boolmask_to_new_DataArray(bmask)

    def __le__(self, other) -> "DataArray":  # type: ignore
        bmask = self.values <= self._other_to_values(other)
        return self._boolmask_to_new_DataArray(bmask)

    def __ge__(self, other) -> "DataArray":  # type: ignore
        bmask = self.values >= self._other_to_values(other)
        return self._boolmask_to_new_DataArray(bmask)

    def __eq__(self, other) -> "DataArray":  # type: ignore
        bmask = self.values == self._other_to_values(other)
        return self._boolmask_to_new_DataArray(bmask)

    def __ne__(self, other) -> "DataArray":  # type: ignore
        bmask = self.values != self._other_to_values(other)
        return self._boolmask_to_new_DataArray(bmask)

    @staticmethod
    def _other_to_values(
        other: "DataArray" | np.ndarray,
    ) -> np.ndarray:
        return other.values if isinstance(other, DataArray) else other

    def _boolmask_to_new_DataArray(self, bmask) -> "DataArray":  # type: ignore
        return DataArray(
            data=bmask,
            time=self.time,
            item=ItemInfo("Boolean"),
            geometry=self.geometry,
            zn=self._zn,
            dt=self._dt,
        )

    # ============= output methods: to_xxx() ===========

    def _to_dataset(self) -> "Dataset":
        """Create a single-item dataset."""
        from mikeio import Dataset

        return Dataset(
            {self.name: self}
        )  # Single-item Dataset (All info is contained in the DataArray, no need for additional info)

    def to_dfs(self, filename: str | Path, **kwargs: Any) -> None:
        """Write data to a new dfs file.

        Parameters
        ----------
        filename: str
            full path to the new dfs file
        dtype: str, np.dtype, DfsSimpleType, optional
            Dfs0 only: set the dfs data type of the written data
            to e.g. np.float64, by default: DfsSimpleType.Float (=np.float32)
        **kwargs: Any
            additional arguments passed to the writing function, e.g. dtype for dfs0

        """
        self._to_dataset().to_dfs(filename, **kwargs)

    def to_dataframe(
        self, *, unit_in_name: bool = False, round_time: str | bool = "ms"
    ) -> pd.DataFrame:
        """Convert to DataFrame.

        Parameters
        ----------
        unit_in_name: bool, optional
            include unit in column name, default False,
        round_time: str, bool, optional
            round time to, by default "ms", use False to avoid rounding

        Returns
        -------
        pd.DataFrame

        """
        return self._to_dataset().to_dataframe(
            unit_in_name=unit_in_name, round_time=round_time
        )

    def to_pandas(self) -> pd.Series:
        """Convert to Pandas Series.

        Returns
        -------
        pd.Series

        """
        return pd.Series(data=self.to_numpy(), index=self.time, name=self.name)

    def to_xarray(self) -> "xarray.DataArray":
        """Export to xarray.DataArray."""
        import xarray as xr

        coords: MutableMapping[str, Any] = {}
        if self._has_time_axis:
            coords["time"] = xr.DataArray(self.time, dims="time")

        if isinstance(self.geometry, Grid1D):
            coords["x"] = xr.DataArray(data=self.geometry.x, dims="x")
        elif isinstance(self.geometry, Grid2D):
            coords["y"] = xr.DataArray(data=self.geometry.y, dims="y")
            coords["x"] = xr.DataArray(data=self.geometry.x, dims="x")
        elif isinstance(self.geometry, Grid3D):
            coords["z"] = xr.DataArray(data=self.geometry.z, dims="z")
            coords["y"] = xr.DataArray(data=self.geometry.y, dims="y")
            coords["x"] = xr.DataArray(data=self.geometry.x, dims="x")
        elif isinstance(self.geometry, GeometryFM2D):
            coords["element"] = xr.DataArray(
                data=self.geometry.element_ids, dims="element"
            )
        elif isinstance(self.geometry, GeometryPoint2D):
            coords["x"] = self.geometry.x
            coords["y"] = self.geometry.y
        elif isinstance(self.geometry, GeometryPoint3D):
            coords["x"] = self.geometry.x
            coords["y"] = self.geometry.y
            coords["z"] = self.geometry.z

        xr_da = xr.DataArray(
            data=self.values,
            name=self.name,
            dims=self.dims,
            coords=coords,
            attrs={
                "name": self.name,
                "units": self.unit.name,
                "eumType": self.type,
                "eumUnit": self.unit,
            },
        )
        return xr_da

    # ===============================================

    def __repr__(self) -> str:
        out = ["<mikeio.DataArray>"]
        if self.name is not None:
            out.append(f"name: {self.name}")

        rest = [
            self._dims_txt(),
            self._time_txt(),
            self._geometry_txt(),
            self._values_txt(),
        ]
        out = out + [x for x in rest if x is not None]

        return "\n".join(out)

    def _dims_txt(self) -> str:
        dims = [f"{self.dims[i]}:{self.shape[i]}" for i in range(self.ndim)]
        dimsstr = ", ".join(dims)
        return f"dims: ({dimsstr})"

    def _time_txt(self) -> str:
        noneq_txt = "" if self.is_equidistant else " non-equidistant"
        timetxt = (
            f"time: {str(self.time[0])} (time-invariant)"
            if self.n_timesteps == 1
            else f"time: {str(self.time[0])} - {str(self.time[-1])} ({self.n_timesteps}{noneq_txt} records)"
        )
        return timetxt

    def _geometry_txt(self) -> str:
        return f"geometry: {self.geometry}"

    def _values_txt(self) -> str:
        if self.ndim == 0 or (self.ndim == 1 and len(self.values) == 1):
            return f"values: {self.values}"
        elif self.ndim == 1 and len(self.values) < 5:
            valtxt = ", ".join([f"{v:0.4g}" for v in self.values])
            return f"values: [{valtxt}]"
        elif self.ndim == 1:
            return f"values: [{self.values[0]:0.4g}, {self.values[1]:0.4g}, ..., {self.values[-1]:0.4g}]"
        else:
            return ""  # raise NotImplementedError()

    @staticmethod
    def _parse_interp_time(
        old_time: pd.DatetimeIndex, new_time: Any
    ) -> pd.DatetimeIndex:
        if isinstance(new_time, pd.DatetimeIndex):
            t_out_index = new_time
        elif hasattr(new_time, "time"):
            t_out_index = pd.DatetimeIndex(new_time.time)
        else:
            # offset = pd.tseries.offsets.DateOffset(seconds=new_time) # This seems identical, but doesn't work with slicing
            offset = pd.Timedelta(seconds=new_time)
            t_out_index = pd.date_range(
                start=old_time[0], end=old_time[-1], freq=offset
            )

        return t_out_index

    @staticmethod
    def _time_by_agg_axis(
        time: pd.DatetimeIndex, axis: int | Sequence[int]
    ) -> pd.DatetimeIndex:
        """New DatetimeIndex after aggregating over time axis."""
        if axis == 0 or (isinstance(axis, Sequence) and 0 in axis):
            time = pd.DatetimeIndex([time[0]])

        return time

    @staticmethod
    def _get_time_idx_list(
        time: pd.DatetimeIndex,
        steps: int | Iterable[int] | str | datetime | pd.DatetimeIndex | slice,
    ) -> list[int] | slice:
        """Find list of idx in DatetimeIndex."""
        return _get_time_idx_list(time, steps)

    @staticmethod
    def _n_selected_timesteps(time: Sized, k: slice | Sized) -> int:
        return _n_selected_timesteps(time, k)

    @staticmethod
    def _is_boolean_mask(x: Any) -> bool:
        if hasattr(x, "dtype"):  # isinstance(x, (np.ndarray, DataArray)):
            return x.dtype == np.dtype("bool")
        return False

    @staticmethod
    def _get_by_boolean_mask(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if data.shape != mask.shape:
            return data[np.broadcast_to(mask, data.shape)]
        return data[mask]

    @staticmethod
    def _set_by_boolean_mask(
        data: np.ndarray, mask: np.ndarray, value: np.ndarray
    ) -> None:
        if data.shape != mask.shape:
            data[np.broadcast_to(mask, data.shape)] = value
        else:
            data[mask] = value

    @staticmethod
    def _parse_time(time: Any) -> pd.DatetimeIndex:
        """Allow anything that we can create a DatetimeIndex from."""
        if time is None:
            time = [pd.Timestamp(2018, 1, 1)]  # TODO is this the correct epoch?
        if isinstance(time, str) or (not isinstance(time, Iterable)):
            time = [time]

        if not isinstance(time, pd.DatetimeIndex):
            index = pd.DatetimeIndex(time)
        else:
            index = time

        if not index.is_monotonic_increasing:
            raise ValueError(
                "Time must be monotonic increasing (only equal or increasing) instances."
            )
        assert isinstance(index, pd.DatetimeIndex)
        return index

    @staticmethod
    def _parse_axis(
        data_shape: tuple[int, ...],
        dims: tuple[str, ...],
        axis: int | tuple[int, ...] | str | None,
    ) -> int | tuple[int, ...]:
        # TODO change to return tuple always
        # axis = 0 if axis == "time" else axis
        if (axis == "spatial") or (axis == "space"):
            if len(data_shape) == 1:
                if dims[0][0] == "t":
                    raise ValueError(f"space axis cannot be selected from dims {dims}")
                return 0
            if "frequency" in dims or "directions" in dims:
                space_name = "node" if "node" in dims else "element"
                return dims.index(space_name)
            else:
                axis = 1 if (len(data_shape) == 2) else tuple(range(1, len(data_shape)))
        if axis is None:
            axis = 0 if (len(data_shape) == 1) else tuple(range(0, len(data_shape)))

        if isinstance(axis, str):
            axis = "time" if axis == "t" else axis
            if axis in dims:
                return dims.index(axis)
            else:
                raise ValueError(
                    f"axis argument '{axis}' not supported! Must be None, int, list of int or 'time' or 'space'"
                )

        return axis
