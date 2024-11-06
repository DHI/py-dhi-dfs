"""Generic functions for working with all types of dfs files."""

from __future__ import annotations
import math
import os
import pathlib
from copy import deepcopy
from datetime import datetime, timedelta
from shutil import copyfile
from collections.abc import Iterable, Sequence
from typing import Union


import numpy as np
import pandas as pd
from mikecore.DfsBuilder import DfsBuilder
from mikecore.DfsFile import (
    DfsDynamicItemInfo,
    DfsFile,
    DfsEqTimeAxis,
    DfsNonEqTimeAxis,
    DfsEqCalendarAxis,
    DfsNonEqCalendarAxis,
)
from mikecore.DfsFileFactory import DfsFileFactory
from mikecore.eum import eumQuantity
from tqdm import tqdm, trange

from . import __dfs_version__
from .dfs._dfs import _get_item_info, _valid_item_numbers
from .eum import ItemInfo
import mikeio


TimeAxis = Union[
    DfsEqTimeAxis, DfsNonEqTimeAxis, DfsEqCalendarAxis, DfsNonEqCalendarAxis
]

show_progress = True


class _ChunkInfo:
    """Class for keeping track of an chunked processing.

    Parameters
    ----------
    n_data : int
        number of data points
    n_chunks : int
        number of chunks

    Attributes
    ----------
    n_data : int
        number of data points
    n_chunks : int
        number of chunks
    chunk_size : int
        number of data points per chunk

    Methods
    -------
    stop(start)
        Return the stop index for a chunk
    chunk_end(start)
        Return the end index for a chunk
    from_dfs(dfs, item_numbers, buffer_size)
        Calculate chunk info based on # of elements in dfs file and selected buffer size

    """

    def __init__(self, n_data: int, n_chunks: int):
        self.n_data = n_data
        self.n_chunks = n_chunks

    def __repr__(self) -> str:
        return f"_ChunkInfo(n_chunks={self.n_chunks}, n_data={self.n_data}, chunk_size={self.chunk_size})"

    @property
    def chunk_size(self) -> int:
        """number of data points per chunk."""
        return math.ceil(self.n_data / self.n_chunks)

    def stop(self, start: int) -> int:
        """Return the stop index for a chunk."""
        return min(start + self.chunk_size, self.n_data)

    def chunk_end(self, start: int) -> int:
        """Return the end index for a chunk."""
        e2 = self.stop(start)
        return self.chunk_size - ((start + self.chunk_size) - e2)

    @staticmethod
    def from_dfs(
        dfs: DfsFile, item_numbers: list[int], buffer_size: float
    ) -> "_ChunkInfo":
        """Calculate chunk info based on # of elements in dfs file and selected buffer size."""
        n_time_steps = dfs.FileInfo.TimeAxis.NumberOfTimeSteps
        n_data_all: int = np.sum([dfs.ItemInfo[i].ElementCount for i in item_numbers])
        mem_need = 8 * n_time_steps * n_data_all  # n_items *
        n_chunks = math.ceil(mem_need / buffer_size)
        n_data = n_data_all // len(item_numbers)

        return _ChunkInfo(n_data, n_chunks)


def _clone(
    infilename: str | pathlib.Path,
    outfilename: str | pathlib.Path,
    start_time: datetime | None = None,
    timestep: float | None = None,
    items: Sequence[int | str | DfsDynamicItemInfo] | None = None,
) -> DfsFile:
    """Clone a dfs file.

    Parameters
    ----------
    infilename : str | pathlib.Path
        input filename
    outfilename : str | pathlib.Path
        output filename
    start_time : datetime, optional
        new start time for the new file, by default None
    timestep : float, optional
        new timestep (in seconds) for the new file, by default None
    items : list(int,str,eum.ItemInfo), optional
        list of items for new file, either as a list of
        ItemInfo or a list of str/int referring to original file,
        default: all items from original file

    Returns
    -------
    DfsFile
        MIKE generic dfs file object

    """
    source = DfsFileFactory.DfsGenericOpen(str(infilename))
    fi = source.FileInfo

    builder = DfsBuilder.Create(fi.FileTitle, "mikeio", __dfs_version__)

    # Set up the header
    builder.SetDataType(fi.DataType)
    builder.SetGeographicalProjection(fi.Projection)

    # Copy time axis
    time_axis = fi.TimeAxis
    if start_time is not None:
        time_axis.StartDateTime = start_time
    if timestep is not None:
        time_axis.TimeStep = timestep
    builder.SetTemporalAxis(time_axis)

    builder.SetItemStatisticsType(fi.StatsType)
    builder.DeleteValueByte = fi.DeleteValueByte
    builder.DeleteValueDouble = fi.DeleteValueDouble
    builder.DeleteValueFloat = fi.DeleteValueFloat
    builder.DeleteValueInt = fi.DeleteValueInt
    builder.DeleteValueUnsignedInt = fi.DeleteValueUnsignedInt

    # Copy custom blocks - if any
    for customBlock in fi.CustomBlocks:
        builder.AddCustomBlock(customBlock)

    names = [x.Name for x in source.ItemInfo]
    item_lookup = {name: i for i, name in enumerate(names)}

    if isinstance(items, Iterable) and not isinstance(items, str):
        for item in items:
            if isinstance(item, ItemInfo):
                builder.AddCreateDynamicItem(
                    item.name,
                    eumQuantity.Create(item.type, item.unit),
                    spatialAxis=source.ItemInfo[-1].SpatialAxis,
                )
            elif isinstance(item, DfsDynamicItemInfo):
                builder.AddDynamicItem(item)
            elif isinstance(item, int):
                builder.AddDynamicItem(source.ItemInfo[item])
            elif isinstance(item, str):
                item_no = item_lookup[item]
                builder.AddDynamicItem(source.ItemInfo[item_no])

    elif isinstance(items, (int, str)) or items is None:
        # must be str/int refering to original file (or None)
        item_numbers = _valid_item_numbers(source.ItemInfo, items)
        items = [source.ItemInfo[item] for item in item_numbers]
        for item in items:
            builder.AddDynamicItem(item)
    else:
        raise ValueError("Items of type: {type(items)} is not supported")

    # Create file
    builder.CreateFile(str(outfilename))

    # Copy static items
    while True:
        static_item = source.ReadStaticItemNext()
        if static_item is None:
            break
        builder.AddStaticItem(static_item)

    # Get the file
    file = builder.GetFile()

    source.Close()

    return file


def scale(
    infilename: str | pathlib.Path,
    outfilename: str | pathlib.Path,
    offset: float = 0.0,
    factor: float = 1.0,
    items: Sequence[int | str] | None = None,
) -> None:
    """Apply scaling to any dfs file.

    Parameters
    ----------

    infilename: str | pathlib.Path
        full path to the input file
    outfilename: str | pathlib.Path
        full path to the output file
    offset: float, optional
        value to add to all items, default 0.0
    factor: float, optional
        value to multiply to all items, default 1.0
    items: list[str] or list[int], optional
        Process only selected items, by number (0-based) or name, by default: all

    """
    infilename = str(infilename)
    outfilename = str(outfilename)
    copyfile(infilename, outfilename)
    dfs = DfsFileFactory.DfsGenericOpenEdit(outfilename)

    item_numbers = _valid_item_numbers(dfs.ItemInfo, items)
    n_items = len(item_numbers)

    n_time_steps = dfs.FileInfo.TimeAxis.NumberOfTimeSteps

    deletevalue = dfs.FileInfo.DeleteValueFloat

    for timestep in trange(n_time_steps, disable=not show_progress):
        for item in range(n_items):
            itemdata = dfs.ReadItemTimeStep(item_numbers[item] + 1, timestep)
            time = itemdata.Time
            d = itemdata.Data
            d[d == deletevalue] = np.nan

            outdata = d * factor + offset

            outdata[np.isnan(outdata)] = deletevalue
            darray = outdata.astype(np.float32)

            dfs.WriteItemTimeStep(item_numbers[item] + 1, timestep, time, darray)

    dfs.Close()


def fill_corrupt(
    infilename: str | pathlib.Path,
    outfilename: str | pathlib.Path,
    fill_value: float = np.nan,
    items: Sequence[str | int] | None = None,
) -> None:
    """Replace corrupt (unreadable) data with fill_value, default delete value.

    Parameters
    ----------

    infilename: str | pathlib.Path
        full path to the input file
    outfilename: str | pathlib.Path
        full path to the output file
    fill_value: float, optional
        value to use where data is corrupt, default delete value
    items: list[str] or list[int], optional
        Process only selected items, by number (0-based) or name, by default: all

    """
    dfs_i = DfsFileFactory.DfsGenericOpen(infilename)

    item_numbers = _valid_item_numbers(dfs_i.ItemInfo, items)
    n_items = len(item_numbers)

    dfs = _clone(
        str(infilename),
        str(outfilename),
        items=item_numbers,
    )

    n_time_steps = dfs_i.FileInfo.TimeAxis.NumberOfTimeSteps

    deletevalue = dfs.FileInfo.DeleteValueFloat

    for timestep in trange(n_time_steps, disable=not show_progress):
        for item in range(n_items):
            itemdata = dfs_i.ReadItemTimeStep(item_numbers[item] + 1, timestep)
            if itemdata is not None:
                time = itemdata.Time
                d = itemdata.Data
            else:
                iteminfo: DfsDynamicItemInfo = dfs_i.ItemInfo[item]
                d = np.zeros(iteminfo.ElementCount)
                d[:] = fill_value
                d[d == deletevalue] = np.nan

                # close and re-open file to solve problem with reading
                dfs_i.Close()
                dfs_i = DfsFileFactory.DfsGenericOpen(infilename)

            d[np.isnan(d)] = deletevalue
            darray = d.astype(np.float32)

            dfs.WriteItemTimeStep(item + 1, timestep, time, darray)

    dfs_i.Close()
    dfs.Close()


def sum(
    infilename_a: str | pathlib.Path,
    infilename_b: str | pathlib.Path,
    outfilename: str | pathlib.Path,
) -> None:
    """Sum two dfs files (a+b).

    Parameters
    ----------
    infilename_a: str | pathlib.Path
        full path to the first input file
    infilename_b: str | pathlib.Path
        full path to the second input file
    outfilename: str | pathlib.Path
        full path to the output file

    """
    infilename_a = str(infilename_a)
    infilename_b = str(infilename_b)
    outfilename = str(outfilename)
    copyfile(infilename_a, outfilename)

    dfs_i_a = DfsFileFactory.DfsGenericOpen(infilename_a)
    dfs_i_b = DfsFileFactory.DfsGenericOpen(infilename_b)
    dfs_o = DfsFileFactory.DfsGenericOpenEdit(outfilename)

    deletevalue = dfs_i_a.FileInfo.DeleteValueFloat

    n_time_steps = dfs_i_a.FileInfo.TimeAxis.NumberOfTimeSteps
    n_items = len(dfs_i_a.ItemInfo)
    # TODO Add checks to verify identical structure of file a and b

    for timestep in trange(n_time_steps):
        for item in range(n_items):
            itemdata_a = dfs_i_a.ReadItemTimeStep(item + 1, timestep)
            d_a = itemdata_a.Data
            d_a[d_a == deletevalue] = np.nan

            itemdata_b = dfs_i_b.ReadItemTimeStep(item + 1, timestep)
            d_b = itemdata_b.Data
            d_a[d_a == deletevalue] = np.nan
            time = itemdata_a.Time

            outdata = d_a + d_b

            darray = outdata.astype(np.float32)

            dfs_o.WriteItemTimeStep(item + 1, timestep, time, darray)

    dfs_i_a.Close()
    dfs_i_b.Close()
    dfs_o.Close()


def diff(
    infilename_a: str | pathlib.Path,
    infilename_b: str | pathlib.Path,
    outfilename: str | pathlib.Path,
) -> None:
    """Calculate difference between two dfs files (a-b).

    Parameters
    ----------
    infilename_a: str | pathlib.Path
        full path to the first input file
    infilename_b: str | pathlib.Path
        full path to the second input file
    outfilename: str | pathlib.Path
        full path to the output file

    """
    infilename_a = str(infilename_a)
    infilename_b = str(infilename_b)
    outfilename = str(outfilename)

    copyfile(infilename_a, outfilename)

    dfs_i_a = DfsFileFactory.DfsGenericOpen(infilename_a)
    dfs_i_b = DfsFileFactory.DfsGenericOpen(infilename_b)
    dfs_o = DfsFileFactory.DfsGenericOpenEdit(outfilename)

    deletevalue = dfs_i_a.FileInfo.DeleteValueFloat

    n_time_steps = dfs_i_a.FileInfo.TimeAxis.NumberOfTimeSteps
    n_items = len(dfs_i_a.ItemInfo)
    # TODO Add checks to verify identical structure of file a and b

    for timestep in trange(n_time_steps):
        for item in range(n_items):
            itemdata_a = dfs_i_a.ReadItemTimeStep(item + 1, timestep)
            d_a = itemdata_a.Data
            d_a[d_a == deletevalue] = np.nan

            itemdata_b = dfs_i_b.ReadItemTimeStep(item + 1, timestep)
            d_b = itemdata_b.Data
            d_b[d_b == deletevalue] = np.nan
            time = itemdata_a.Time

            outdata = d_a - d_b

            d = outdata.astype(np.float32)
            d[np.isnan(d)] = deletevalue

            dfs_o.WriteItemTimeStep(item + 1, timestep, time, d)

    dfs_i_a.Close()
    dfs_i_b.Close()
    dfs_o.Close()


def concat(
    infilenames: Sequence[str | pathlib.Path],
    outfilename: str | pathlib.Path,
    keep: str = "last",
) -> None:
    """Concatenates files along the time axis.

    Overlap handling is defined by the `keep` argument,  by default the last one will be used.

    Parameters
    ----------
    infilenames:
        filenames to concatenate
    outfilename:
        filename of output
    keep:
        either 'first' (keep older), 'last' (keep newer)
        or 'average' can be selected. By default 'last'

    Notes
    ------

    The list of input files have to be sorted, i.e. in chronological order

    """
    # fast path for Dfs0
    suffix = pathlib.Path(infilenames[0]).suffix
    if suffix == ".dfs0":
        dss = [mikeio.read(f) for f in infilenames]
        ds = mikeio.Dataset.concat(dss, keep=keep)  # type: ignore
        ds.to_dfs(outfilename)
        return

    dfs_i_a = DfsFileFactory.DfsGenericOpen(str(infilenames[0]))

    dfs_o = _clone(str(infilenames[0]), str(outfilename))

    n_items = len(dfs_i_a.ItemInfo)
    dfs_i_a.Close()

    current_time = datetime(1, 1, 1)  # beginning of time...

    for i, infilename in enumerate(tqdm(infilenames, disable=not show_progress)):
        dfs_i = DfsFileFactory.DfsGenericOpen(str(infilename))
        t_axis = dfs_i.FileInfo.TimeAxis
        n_time_steps = t_axis.NumberOfTimeSteps
        dt = t_axis.TimeStep
        start_time = t_axis.StartDateTime

        if i > 0 and start_time > current_time + timedelta(seconds=dt):
            dfs_o.Close()
            os.remove(outfilename)
            raise Exception("Gap in time axis detected - not supported")

        current_time = start_time

        if keep == "last":
            if i < (len(infilenames) - 1):
                dfs_n = DfsFileFactory.DfsGenericOpen(str(infilenames[i + 1]))
                nf = dfs_n.FileInfo.TimeAxis.StartDateTime
                next_start_time = datetime(
                    nf.year, nf.month, nf.day, nf.hour, nf.minute, nf.second
                )
                dfs_n.Close()

            for timestep in range(n_time_steps):
                current_time = start_time + timedelta(seconds=timestep * dt)
                if i < (len(infilenames) - 1):
                    if current_time >= next_start_time:
                        break

                for item in range(n_items):
                    itemdata = dfs_i.ReadItemTimeStep(item + 1, timestep)
                    d = itemdata.Data

                    darray = d.astype(np.float32)

                    dfs_o.WriteItemTimeStepNext(0, darray)
            dfs_i.Close()

        if keep == "first":
            if (
                i == 0
            ):  # all timesteps in first file are kept (is there a more efficient way to do this without the loop?)
                for timestep in range(n_time_steps):
                    current_time = start_time + timedelta(seconds=timestep * dt)

                    for item in range(n_items):
                        itemdata = dfs_i.ReadItemTimeStep(item + 1, timestep)
                        d = itemdata.Data

                        darray = d.astype(np.float32)

                        dfs_o.WriteItemTimeStepNext(0, darray)
                end_time = start_time + timedelta(
                    seconds=timestep * dt
                )  # reuse last timestep since there is no EndDateTime attribute in t_axis.
                dfs_i.Close()

            else:
                # determine overlap in number of timesteps
                start_timestep = (
                    int((end_time - start_time) / timedelta(seconds=dt)) + 1
                )
                # loop only through those timesteps that are not in previous file
                for timestep in range(start_timestep, n_time_steps):
                    current_time = start_time + timedelta(seconds=timestep * dt)

                    for item in range(n_items):
                        itemdata = dfs_i.ReadItemTimeStep(item + 1, timestep)
                        d = itemdata.Data

                        darray = d.astype(np.float32)

                        dfs_o.WriteItemTimeStepNext(0, darray)
                end_time = start_time + timedelta(
                    seconds=timestep * dt
                )  # get end time from current file
                dfs_i.Close()

        if keep == "average":
            if i == 0:
                # For first file, write all timesteps normally
                for timestep in range(n_time_steps):
                    current_time = start_time + timedelta(seconds=timestep * dt)

                    for item in range(n_items):
                        itemdata = dfs_i.ReadItemTimeStep(int(item + 1), int(timestep))
                        darray = itemdata.Data.astype(np.float32)
                        dfs_o.WriteItemTimeStepNext(0, darray)
                end_time = start_time + timedelta(seconds=(n_time_steps - 1) * dt)
                first_start_time = start_time  # Store the start time of first file
                dfs_i.Close()
            else:
                # For subsequent files, we need to handle overlapping periods
                # Calculate overlap in timesteps
                if start_time <= end_time:

                    overlap_end = int((end_time - start_time).total_seconds() / dt) + 1

                    # Read current file data
                    for timestep in range(n_time_steps):
                        current_time = start_time + timedelta(seconds=timestep * dt)

                        for item in range(n_items):
                            itemdata = dfs_i.ReadItemTimeStep(
                                int(item + 1), int(timestep)
                            )
                            current_data = itemdata.Data.astype(np.float32)

                            if (
                                int(timestep) <= overlap_end
                            ):  # Convert back to int for comparison
                                # In overlapping period
                                existing_pos = int(
                                    (current_time - first_start_time).total_seconds()
                                    / dt
                                    - 1
                                )
                                if existing_pos >= 0:
                                    # Read existing data
                                    dfs_o.Flush()
                                    temp_dfs = DfsFileFactory.DfsGenericOpen(
                                        str(outfilename)
                                    )
                                    existing_data = temp_dfs.ReadItemTimeStep(
                                        int(item + 1), int(existing_pos)
                                    ).Data
                                    temp_dfs.Close()

                                    # Calculate average
                                    averaged_data = (existing_data + current_data) / 2
                                    # Write averaged data
                                    # dfs_o.WriteItemTimeStep(
                                    #     item + 1, existing_pos, 0, averaged_data
                                    # ) # This is what we need, but it does not wo
                                    dfs_o.WriteItemTimeStepNext(0, averaged_data)

                            else:
                                # After overlap period - write new data
                                dfs_o.WriteItemTimeStepNext(0, current_data)

                else:
                    # No overlap - write all timesteps
                    for timestep in range(n_time_steps):
                        current_time = start_time + timedelta(seconds=timestep * dt)
                        for item in range(n_items):
                            itemdata = dfs_i.ReadItemTimeStep(
                                str(item + 1), str(timestep)
                            )
                            darray = itemdata.Data.astype(np.float32)
                            dfs_o.WriteItemTimeStepNext(0, darray)

                end_time = start_time + timedelta(seconds=(n_time_steps - 1) * dt)
                dfs_i.Close()

    dfs_o.Close()


def extract(
    infilename: str | pathlib.Path,
    outfilename: str | pathlib.Path,
    start: int = 0,
    end: int = -1,
    step: int = 1,
    items: Sequence[int | str] | None = None,
) -> None:
    """Extract timesteps and/or items to a new dfs file.

    Parameters
    ----------
    infilename : str | pathlib.Path
        path to input dfs file
    outfilename : str | pathlib.Path
        path to output dfs file
    start : int, float, str or datetime, optional
        start of extraction as either step, relative seconds
        or datetime/str, by default 0 (start of file)
    end : int, float, str or datetime, optional
        end of extraction as either step, relative seconds
        or datetime/str, by default -1 (end of file)
    step : int, optional
        jump this many step, by default 1 (every step between start and end)
    items : int, list(int), str, list(str), optional
        items to be extracted to new file

    Examples
    --------
    >>> extract('f_in.dfs0', 'f_out.dfs0', start='2018-1-1')
    >>> extract('f_in.dfs2', 'f_out.dfs2', end=-3)
    >>> extract('f_in.dfsu', 'f_out.dfsu', start=1800.0, end=3600.0)
    >>> extract('f_hourly.dfsu', 'f_daily.dfsu', step=24)
    >>> extract('f_in.dfsu', 'f_out.dfsu', items=[2, 0])
    >>> extract('f_in.dfsu', 'f_out.dfsu', items="Salinity")
    >>> extract('f_in.dfsu', 'f_out.dfsu', end='2018-2-1 00:00', items="Salinity")

    """
    dfs_i = DfsFileFactory.DfsGenericOpenEdit(str(infilename))

    is_layered_dfsu = dfs_i.ItemInfo[0].Name == "Z coordinate"

    file_start_new, start_step, start_sec, end_step, end_sec = _parse_start_end(
        dfs_i.FileInfo.TimeAxis, start, end
    )
    timestep = _parse_step(dfs_i.FileInfo.TimeAxis, step)
    item_numbers = _valid_item_numbers(
        dfs_i.ItemInfo, items, ignore_first=is_layered_dfsu
    )

    if is_layered_dfsu:
        item_numbers = [it + 1 for it in item_numbers]
        item_numbers.insert(0, 0)

    dfs_o = _clone(
        str(infilename),
        str(outfilename),
        start_time=file_start_new,
        timestep=timestep,
        items=item_numbers,
    )

    file_start_shift = 0
    if file_start_new is not None:
        file_start_orig = dfs_i.FileInfo.TimeAxis.StartDateTime
        file_start_shift = (file_start_new - file_start_orig).total_seconds()

    timestep_out = -1
    for timestep in range(start_step, end_step, step):
        for item_out, item in enumerate(item_numbers):
            itemdata = dfs_i.ReadItemTimeStep((item + 1), timestep)
            time_sec = itemdata.Time

            if time_sec > end_sec:
                dfs_i.Close()
                dfs_o.Close()
                return

            if time_sec >= start_sec:
                if item == item_numbers[0]:
                    timestep_out = timestep_out + 1
                time_sec_out = time_sec - file_start_shift

                outdata = itemdata.Data
                dfs_o.WriteItemTimeStep(
                    (item_out + 1), timestep_out, time_sec_out, outdata
                )

    dfs_i.Close()
    dfs_o.Close()


def _parse_start_end(
    time_axis: TimeAxis,
    start: int | float | str | datetime,
    end: int | float | str | datetime,
) -> tuple[datetime | None, int, float, int, float]:  # TODO better return type
    """Helper function for parsing start and end arguments."""
    n_time_steps = time_axis.NumberOfTimeSteps
    file_start_datetime = time_axis.StartDateTime
    file_start_sec = time_axis.StartTimeOffset
    start_sec = file_start_sec

    timespan = 0
    if time_axis.TimeAxisType == 3:
        timespan = time_axis.TimeStep * (n_time_steps - 1)
    elif time_axis.TimeAxisType == 4:
        timespan = time_axis.TimeSpan
    else:
        raise ValueError("TimeAxisType not supported")

    file_end_sec = start_sec + timespan
    end_sec = file_end_sec

    start_step = 0
    if isinstance(start, int):
        start_step = start
    elif isinstance(start, float):
        start_sec = start
    elif isinstance(start, str):
        parts = start.split(",")
        start = parts[0]
        if len(parts) == 2:
            end = parts[1]
        start = pd.to_datetime(start)

    if isinstance(start, datetime):
        start_sec = (start - file_start_datetime).total_seconds()

    end_step = n_time_steps
    if isinstance(end, int):
        if end < 0:
            end = end_step + end + 1
        end_step = end
    elif isinstance(end, float):
        end_sec = end
    elif isinstance(end, str):
        end = pd.to_datetime(end)

    if isinstance(end, datetime):
        end_sec = (end - file_start_datetime).total_seconds()

    if start_step < 0:
        raise ValueError(
            f"start cannot be before start of file. start={start_step} is invalid"
        )

    if start_sec < file_start_sec:
        raise ValueError(
            f"start cannot be before start of file start={start_step} is invalid"
        )

    if (end_sec < start_sec) or (end_step < start_step):
        raise ValueError("end must be after start")

    if end_step > n_time_steps:
        raise ValueError(f"end cannot be after end of file. end={end_step} is invalid.")

    if end_sec > file_end_sec:
        raise ValueError(f"end cannot be after end of file. end={end_sec} is invalid.")

    file_start_new = None
    if time_axis.TimeAxisType == 3:
        dt = time_axis.TimeStep
        if (start_sec > file_start_sec) and (start_step == 0):
            # we can find the coresponding step
            start_step = int((start_sec - file_start_sec) / dt)
        file_start_new = file_start_datetime + timedelta(seconds=start_step * dt)
    elif time_axis.TimeAxisType == 4:
        if start_sec > file_start_sec:
            file_start_new = file_start_datetime + timedelta(seconds=start_sec)

    return file_start_new, start_step, start_sec, end_step, end_sec


def _parse_step(time_axis: TimeAxis, step: int) -> float | None:
    """Helper function for parsing step argument."""
    if step == 1:
        timestep = None
    elif time_axis.TimeAxisType == 3:
        timestep = time_axis.TimeStep * step
    elif time_axis.TimeAxisType == 4:
        timestep = None
    else:
        raise ValueError("TimeAxisType not supported")
    return timestep


def avg_time(
    infilename: str | pathlib.Path,
    outfilename: str | pathlib.Path,
    skipna: bool = True,
) -> None:
    """Create a temporally averaged dfs file.

    Parameters
    ----------
    infilename : str | pathlib.Path
        input filename
    outfilename : str | pathlib.Path
        output filename
    skipna : bool, optional
        exclude NaN/delete values when computing the result, default True

    """
    dfs_i = DfsFileFactory.DfsGenericOpen(str(infilename))

    dfs_o = _clone(infilename, outfilename)

    n_items = len(dfs_i.ItemInfo)
    item_numbers = list(range(n_items))

    n_time_steps = dfs_i.FileInfo.TimeAxis.NumberOfTimeSteps
    deletevalue = dfs_i.FileInfo.DeleteValueFloat

    outdatalist = []
    steps_list = []

    # step 0
    for item in item_numbers:
        indatatime = dfs_i.ReadItemTimeStep(item + 1, 0.0)
        indata = indatatime.Data
        has_value = indata != deletevalue
        indata[~has_value] = np.nan
        outdatalist.append(indata.astype(np.float32))
        step0 = np.zeros_like(indata, dtype=np.int32)
        step0[has_value] = 1
        steps_list.append(step0)

    for timestep in trange(1, n_time_steps, disable=not show_progress):
        for item in range(n_items):
            itemdata = dfs_i.ReadItemTimeStep(item_numbers[item] + 1, timestep)
            d = itemdata.Data
            has_value = d != deletevalue

            outdatalist[item][has_value] += d[has_value]
            steps_list[item][has_value] += 1

    for item in range(n_items):
        darray = np.zeros_like(outdatalist[item], dtype=np.float32)
        if skipna:
            has_value = steps_list[item] == n_time_steps
        else:
            has_value = steps_list[item] > 0
        darray[has_value] = outdatalist[item][has_value].astype(
            np.float32
        ) / steps_list[item][has_value].astype(np.float32)
        darray[~has_value] = deletevalue
        dfs_o.WriteItemTimeStepNext(0.0, darray)

    dfs_o.Close()


def quantile(
    infilename: str | pathlib.Path,
    outfilename: str | pathlib.Path,
    q: float | Sequence[float],
    *,
    items: Sequence[int | str] | None = None,
    skipna: bool = True,
    buffer_size: float = 1.0e9,
) -> None:
    """Create temporal quantiles of all items in dfs file.

    Parameters
    ----------
    infilename : str | pathlib.Path
        input filename
    outfilename : str | pathlib.Path
        output filename
    q: array_like of float
        Quantile or sequence of quantiles to compute,
        which must be between 0 and 1 inclusive.
    items: list[str] or list[int], optional
        Process only selected items, by number (0-based) or name, by default: all
    skipna : bool, optional
        exclude NaN/delete values when computing the result, default True
    buffer_size: float, optional
        for huge files the quantiles need to be calculated for chunks of
        elements. buffer_size gives the maximum amount of memory available
        for the computation in bytes, by default 1e9 (=1GB)

    Examples
    --------
    >>> quantile("in.dfsu", "IQR.dfsu", q=[0.25,0.75])

    >>> quantile("huge.dfsu", "Q01.dfsu", q=0.1, buffer_size=5.0e9)

    >>> quantile("with_nans.dfsu", "Q05.dfsu", q=0.5, skipna=False)

    """
    func = np.nanquantile if skipna else np.quantile

    dfs_i = DfsFileFactory.DfsGenericOpen(infilename)

    is_dfsu_3d = dfs_i.ItemInfo[0].Name == "Z coordinate"

    item_numbers = _valid_item_numbers(dfs_i.ItemInfo, items)

    if is_dfsu_3d and 0 in item_numbers:
        item_numbers.remove(0)  # Remove Zn item for special treatment

    n_items_in = len(item_numbers)

    n_time_steps = dfs_i.FileInfo.TimeAxis.NumberOfTimeSteps

    # TODO: better handling of different item sizes (zn...)

    ci = _ChunkInfo.from_dfs(dfs_i, item_numbers, buffer_size)

    qvec: Sequence[float] = [q] if isinstance(q, float) else q
    qtxt = [f"Quantile {q!r}" for q in qvec]
    core_items = [dfs_i.ItemInfo[i] for i in item_numbers]
    items = _get_repeated_items(core_items, prefixes=qtxt)

    if is_dfsu_3d:
        items.insert(0, dfs_i.ItemInfo[0])

    dfs_o = _clone(infilename, outfilename, items=items)

    n_items_out = len(items)
    if is_dfsu_3d:
        n_items_out = n_items_out - 1

    datalist = []
    outdatalist = []

    for item in item_numbers:
        indata = _read_item(dfs_i, item, 0)
        for _ in qvec:
            outdatalist.append(np.zeros_like(indata))
        datalist.append(np.zeros((n_time_steps, ci.chunk_size)))

    e1 = 0
    for _ in range(ci.n_chunks):
        e2 = ci.stop(e1)
        # the last chunk may be smaller than the rest:
        chunk_end = ci.chunk_end(e1)

        # read all data for this chunk
        for timestep in range(n_time_steps):
            item_out = 0
            for item_no in item_numbers:
                itemdata = _read_item(dfs_i, item_no, timestep)
                data_chunk = itemdata[e1:e2]
                datalist[item_out][timestep, 0:chunk_end] = data_chunk
                item_out += 1

        # calculate quantiles (for this chunk)
        item_out = 0
        for item in range(n_items_in):
            qdat = np.zeros((len(qvec), (ci.chunk_size)))
            qdat[:, :] = func(datalist[item][:, 0:chunk_end], q=qvec, axis=0)
            for j in range(len(qvec)):
                outdatalist[item_out][e1:e2] = qdat[j, :]
                item_out += 1

        e1 = e2

    if is_dfsu_3d:
        znitemdata = dfs_i.ReadItemTimeStep(1, 0)
        # TODO should this be static Z coordinates instead?
        dfs_o.WriteItemTimeStepNext(0.0, znitemdata.Data)

    for item in range(n_items_out):
        darray = outdatalist[item].astype(np.float32)
        dfs_o.WriteItemTimeStepNext(0.0, darray)

    dfs_o.Close()


def _read_item(dfs: DfsFile, item: int, timestep: int) -> np.ndarray:
    """Read item data from dfs file.

    Parameters
    ----------
    dfs : DfsFile
        dfs file
    item : int
        item number
    timestep : int
        timestep number

    Returns
    -------
    np.ndarray
        item data

    """
    indatatime = dfs.ReadItemTimeStep(item + 1, timestepIndex=timestep)
    indata = indatatime.Data
    has_value = indata != dfs.FileInfo.DeleteValueFloat
    indata[~has_value] = np.nan

    return indata.astype(np.float64)


def _get_repeated_items(
    items_in: list[DfsDynamicItemInfo], prefixes: list[str]
) -> list[ItemInfo]:
    """Create new items by repeating the items in items_in with the prefixes.

    Parameters
    ----------
    items_in : list[DfsDynamicItemInfo]
        List of items to be repeated
    prefixes : list[str]
        List of prefixes to be added to the items

    Returns
    -------
    list[ItemInfo]
        List of new items

    """
    item_numbers = _valid_item_numbers(items_in)
    items_in = _get_item_info(items_in)

    new_items = []
    for item_num in item_numbers:
        for prefix in prefixes:
            item = deepcopy(items_in[item_num])
            item.name = f"{prefix}, {item.name}"
            new_items.append(item)

    return new_items
