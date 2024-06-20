from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from typing import Any, List, Literal, Mapping, Tuple
from collections.abc import Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from mikecore.DfsFactory import DfsBuilder, DfsFactory
from mikecore.DfsFile import DfsFile, DfsSimpleType
from mikecore.DfsFileFactory import DfsFileFactory
from mikecore.eum import eumQuantity, eumUnit
from mikecore.Projections import Cartography

from .. import __dfs_version__
from ..dataset import Dataset
from ._dfs import (
    _Dfs123,
    _get_item_info,
    _valid_item_numbers,
    _valid_timesteps,
    write_dfs_data,
)
from ..eum import TimeStepUnit
from ..spatial import Grid2D


def write_dfs2(
    filename: str | Path,
    ds: Dataset,
    title: str = "",
    custom_blocks: Mapping[str, np.ndarray] | None = None,
) -> None:
    """Write a dfs2 file

    Parameters
    ----------
    filename: str or Path
        File name to write
    ds: Dataset
        The dataset to write
    title: str, optional
        Title of the file
    custom_blocks: dict, optional
        Custom blocks to add to the file
    """

    dfs = _write_dfs2_header(filename, ds, title, custom_blocks=custom_blocks)
    write_dfs_data(dfs=dfs, ds=ds, n_spatial_dims=2)


def _write_dfs2_header(
    filename: str | Path,
    ds: Dataset,
    title: str = "",
    custom_blocks: Mapping[str, np.ndarray] | None = None,
) -> DfsFile:
    builder = DfsBuilder.Create(title, "mikeio", __dfs_version__)
    builder.SetDataType(0)
    if custom_blocks is not None:
        factory = DfsFactory()
        for key, value in custom_blocks.items():
            builder.AddCustomBlock(factory.CreateCustomBlock(key, value))

    geometry: Grid2D = ds.geometry

    if (
        geometry._shift_origin_on_write
        and not geometry._is_rotated
        and not geometry.is_spectral
    ):
        geometry = deepcopy(ds.geometry)
        geometry._shift_x0y0_to_origin()

    factory = DfsFactory()
    _write_dfs2_spatial_axis(builder, factory, geometry)
    proj_str = geometry.projection_string
    origin = geometry.origin
    orient = geometry.orientation

    if geometry.is_geo:
        proj = factory.CreateProjectionGeoOrigin(proj_str, *origin, orient)
    else:
        cart: Cartography = Cartography.CreateProjOrigin(proj_str, *origin, orient)
        proj = factory.CreateProjectionGeoOrigin(
            wktProjectionString=geometry.projection,
            lon0=cart.LonOrigin,
            lat0=cart.LatOrigin,
            orientation=cart.Orientation,
        )

    builder.SetGeographicalProjection(proj)

    timestep_unit = TimeStepUnit.SECOND
    dt = ds.timestep or 1.0  # It can not be None
    if ds.is_equidistant:
        time_axis = factory.CreateTemporalEqCalendarAxis(
            timestep_unit, ds.time[0], 0, dt
        )
    else:
        time_axis = factory.CreateTemporalNonEqCalendarAxis(timestep_unit, ds.time[0])
    builder.SetTemporalAxis(time_axis)

    for item in ds.items:
        builder.AddCreateDynamicItem(
            item.name,
            eumQuantity.Create(item.type, item.unit),
            DfsSimpleType.Float,
            item.data_value_type,
        )

    try:
        builder.CreateFile(str(filename))
    except IOError:
        print("cannot create dfs file: ", filename)

    return builder.GetFile()


def _write_dfs2_spatial_axis(
    builder: DfsBuilder, factory: DfsFactory, geometry: Grid2D
) -> None:
    builder.SetSpatialAxis(
        factory.CreateAxisEqD2(
            eumUnit.eumUmeter,
            geometry._nx,
            geometry._x0,
            geometry._dx,
            geometry._ny,
            geometry._y0,
            geometry._dy,
        )
    )


class Dfs2(_Dfs123):
    _ndim = 2

    def __init__(
        self,
        filename: str | Path,
        type: Literal["horizontal", "spectral", "vertical"] = "horizontal",
    ):
        filename = str(filename)
        super().__init__(filename)

        is_spectral = type == "spectral"
        is_vertical = type == "vertical"
        dfs = DfsFileFactory.Dfs2FileOpen(str(filename))

        x0 = dfs.SpatialAxis.X0 if is_spectral else 0.0
        y0 = dfs.SpatialAxis.Y0 if is_spectral else 0.0

        origin, orientation = self._origin_and_orientation_in_CRS()

        self._geometry = Grid2D(
            dx=dfs.SpatialAxis.Dx,
            dy=dfs.SpatialAxis.Dy,
            nx=dfs.SpatialAxis.XCount,
            ny=dfs.SpatialAxis.YCount,
            x0=x0,
            y0=y0,
            projection=self._projstr,
            orientation=orientation,
            origin=origin,
            is_spectral=is_spectral,
            is_vertical=is_vertical,
        )
        dfs.Close()
        self._validate_no_orientation_in_geo()

    def read(
        self,
        *,
        items: str | int | Sequence[str | int] | None = None,
        time: int | str | slice | None = None,
        area: Tuple[float, float, float, float] | None = None,
        keepdims: bool = False,
        dtype: Any = np.float32,
    ) -> Dataset:
        """
        Read data from a dfs2 file

        Parameters
        ---------
        items: list[int] or list[str], optional
            Read only selected items, by number (0-based), or by name
        time: int, str, datetime, pd.TimeStamp, sequence, slice or pd.DatetimeIndex, optional
            Read only selected time steps, by default None (=all)
        keepdims: bool, optional
            When reading a single time step only, should the time-dimension be kept
            in the returned Dataset? by default: False
        area: array[float], optional
            Read only data inside (horizontal) area given as a
            bounding box (tuple with left, lower, right, upper) coordinates
        dtype: data-type, optional
            Define the dtype of the returned dataset (default = np.float32)
        Returns
        -------
        Dataset
        """

        self._open()

        item_numbers = _valid_item_numbers(self._dfs.ItemInfo, items)
        n_items = len(item_numbers)
        items = _get_item_info(self._dfs.ItemInfo, item_numbers)

        single_time_selected, time_steps = _valid_timesteps(self._dfs.FileInfo, time)
        nt = len(time_steps) if not single_time_selected else 1

        shape: Tuple[int, ...]

        if area is not None:
            take_subset = True
            ii, jj = self.geometry.find_index(area=area)  # type: ignore
            shape = (nt, len(jj), len(ii))
            geometry = self.geometry._index_to_Grid2D(ii, jj)
        else:
            take_subset = False
            shape = (nt, self.ny, self.nx)
            geometry = self.geometry

        if single_time_selected and not keepdims:
            shape = shape[1:]

        data_list: List[np.ndarray] = [
            np.ndarray(shape=shape, dtype=dtype) for _ in range(n_items)
        ]

        t_seconds = np.zeros(len(time_steps))

        for i, it in enumerate(tqdm(time_steps, disable=not self.show_progress)):
            for item in range(n_items):
                itemdata = self._dfs.ReadItemTimeStep(item_numbers[item] + 1, int(it))
                d = itemdata.Data

                d[d == self.deletevalue] = np.nan
                d = d.reshape(self.ny, self.nx)

                if take_subset:
                    d = np.take(np.take(d, jj, axis=0), ii, axis=-1)

                if single_time_selected and not keepdims:
                    data_list[item] = d
                else:
                    data_list[item][i] = d

            t_seconds[i] = itemdata.Time

        self._dfs.Close()

        time = pd.to_datetime(t_seconds, unit="s", origin=self.start_time)

        dims: Tuple[str, ...]

        if single_time_selected and not keepdims:
            dims = ("y", "x")
        else:
            dims = ("time", "y", "x")

        return Dataset(
            data_list,
            time=time,
            items=items,
            geometry=geometry,
            dims=dims,
            validate=False,
        )

    def _open(self) -> None:
        self._dfs = DfsFileFactory.Dfs2FileOpen(self._filename)
        self._source = self._dfs

    @property
    def geometry(self) -> Grid2D:
        """Spatial information"""
        assert isinstance(self._geometry, Grid2D)
        return self._geometry

    @property
    def x0(self) -> Any:
        """Start point of x values (often 0)"""
        return self.geometry.x[0]

    @property
    def y0(self) -> Any:
        """Start point of y values (often 0)"""
        return self.geometry.y[0]

    @property
    def dx(self) -> float:
        """Step size in x direction"""
        return self.geometry.dx

    @property
    def dy(self) -> float:
        """Step size in y direction"""
        return self.geometry.dy

    @property
    def shape(self) -> Tuple[int, ...]:
        """Tuple with number of values in the t-, y-, x-direction"""
        return (self._n_timesteps, self.geometry.ny, self.geometry.nx)

    @property
    def nx(self) -> int:
        """Number of values in the x-direction"""
        return self.geometry.nx

    @property
    def ny(self) -> int:
        """Number of values in the y-direction"""
        return self.geometry.ny
