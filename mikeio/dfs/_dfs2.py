from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple
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
    _write_dfs_data,
)
from ..eum import TimeStepUnit
from ..spatial import Grid2D


def write_dfs2(filename: str | Path, ds: Dataset, title: str = "") -> None:
    dfs = _write_dfs2_header(filename, ds, title)
    _write_dfs_data(dfs=dfs, ds=ds, n_spatial_dims=2)


def _write_dfs2_header(filename: str | Path, ds: Dataset, title: str = "") -> DfsFile:
    builder = DfsBuilder.Create(title, "mikeio", __dfs_version__)
    builder.SetDataType(0)

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


def _write_dfs2_spatial_axis(builder, factory, geometry):
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

    def __init__(self, filename: str | Path, type: str = "horizontal"):
        filename = str(filename)
        super().__init__(filename)

        # TODO move to base class
        self._read_header(filename)

        # TODO
        self._x0 = 0.0
        self._y0 = 0.0

        is_spectral = type.lower() in ["spectral", "spectra", "spectrum"]
        # self._read_dfs2_header(filename=filename, read_x0y0=is_spectral)
        self._geometry = self._read_geometry(filename=filename, is_spectral=is_spectral)
        self._validate_no_orientation_in_geo()
        # origin, orientation = self._origin_and_orientation_in_CRS()

        # self.geometry = Grid2D(
        #     dx=self._dx,
        #     dy=self._dy,
        #     nx=self._nx,
        #     ny=self._ny,
        #     x0=self._x0,
        #     y0=self._y0,
        #     orientation=orientation,
        #     origin=origin,
        #     projection=self._projstr,
        #     is_spectral=is_spectral,
        # )

    def __repr__(self):
        out = ["<mikeio.Dfs2>"]

        if self._filename:
            out.append(f"dx: {self.dx:.5f}")
            out.append(f"dy: {self.dy:.5f}")

            if self._n_items is not None:
                if self._n_items < 10:
                    out.append("items:")
                    for i, item in enumerate(self.items):
                        out.append(f"  {i}:  {item}")
                else:
                    out.append(f"number of items: {self._n_items}")

                if self._n_timesteps == 1:
                    out.append("time: time-invariant file (1 step)")
                else:
                    out.append(f"time: {self._n_timesteps} steps")
                    out.append(f"start time: {self._start_time}")

        return str.join("\n", out)

    def _read_geometry(self, filename: str | Path, is_spectral: bool = False) -> Grid2D:
        dfs = DfsFileFactory.Dfs2FileOpen(str(filename))

        x0 = dfs.SpatialAxis.X0 if is_spectral else 0.0
        y0 = dfs.SpatialAxis.Y0 if is_spectral else 0.0

        origin, orientation = self._origin_and_orientation_in_CRS()

        geometry = Grid2D(
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
        )
        dfs.Close()
        return geometry

    def _read_dfs2_header(self, filename: str | Path, read_x0y0: bool = False) -> None:
        self._dfs = DfsFileFactory.Dfs2FileOpen(str(filename))
        self._source = self._dfs
        if read_x0y0:
            self._x0 = self._dfs.SpatialAxis.X0
            self._y0 = self._dfs.SpatialAxis.Y0
        self._dx = self._dfs.SpatialAxis.Dx
        self._dy = self._dfs.SpatialAxis.Dy
        self._nx = self._dfs.SpatialAxis.XCount
        self._ny = self._dfs.SpatialAxis.YCount
        if self._dfs.FileInfo.TimeAxis.TimeAxisType == 4:
            self._is_equidistant = False

        self._read_header()

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
            ii, jj = self.geometry.find_index(area=area)
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

        time = pd.to_datetime(t_seconds, unit="s", origin=self.start_time)  # type: ignore

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

    def _open(self):
        self._dfs = DfsFileFactory.Dfs2FileOpen(self._filename)
        self._source = self._dfs

    def _set_spatial_axis(self):
        self._builder.SetSpatialAxis(
            self._factory.CreateAxisEqD2(
                eumUnit.eumUmeter,
                self._nx,
                self._x0,
                self._dx,
                self._ny,
                self._y0,
                self._dy,
            )
        )

    @property
    def geometry(self) -> Grid2D:
        """Spatial information"""
        return self._geometry

    @property
    def x0(self):
        """Start point of x values (often 0)"""
        return self.geometry.x[0]

    @property
    def y0(self):
        """Start point of y values (often 0)"""
        return self.geometry.y[0]

    @property
    def dx(self):
        """Step size in x direction"""
        return self.geometry.dx

    @property
    def dy(self):
        """Step size in y direction"""
        return self.geometry.dy

    @property
    def shape(self) -> Tuple[int, ...]:
        """Tuple with number of values in the t-, y-, x-direction"""
        return (self._n_timesteps, self.geometry.ny, self.geometry.nx)

    @property
    def nx(self):
        """Number of values in the x-direction"""
        return self.geometry.nx

    @property
    def ny(self):
        """Number of values in the y-direction"""
        return self.geometry.ny

    @property
    def is_geo(self):
        """Are coordinates geographical (LONG/LAT)?"""
        return self._projstr == "LONG/LAT"
