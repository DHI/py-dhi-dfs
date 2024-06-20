from pathlib import Path
import datetime
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray

import mikeio

from mikeio import EUMType, ItemInfo, EUMUnit
from mikeio.exceptions import ItemsError
from mikeio.spatial import GeometryPoint2D, Grid2D


@pytest.fixture
def dfs2_random():
    filepath = Path("tests/testdata/random.dfs2")
    return mikeio.open(filepath)


@pytest.fixture
def dfs2_random_2items():
    filepath = Path("tests/testdata/random_two_item.dfs2")
    return mikeio.open(filepath)


@pytest.fixture
def dfs2_pt_spectrum():
    filepath = Path("tests/testdata/pt_spectra.dfs2")
    return mikeio.open(filepath, type="spectral")


@pytest.fixture
def dfs2_pt_spectrum_linearf():
    filepath = Path("tests/testdata/dir_wave_analysis_spectra.dfs2")
    return mikeio.open(filepath, type="spectral")


@pytest.fixture
def dfs2_vertical_nonutm():
    filepath = Path("tests/testdata/hd_vertical_slice.dfs2")
    return mikeio.open(filepath, type="vertical")


@pytest.fixture
def dfs2_gebco():
    filepath = Path("tests/testdata/gebco_sound.dfs2")
    return mikeio.open(filepath)


def test_get_time_without_reading_data():
    dfs = mikeio.open("tests/testdata/hd_vertical_slice.dfs2", type="vertical")

    assert isinstance(dfs.time, pd.DatetimeIndex)
    assert len(dfs.time) == 13
    assert dfs.time[-1].hour == 12


def test_write_projected(tmp_path):

    fp = tmp_path / "utm.dfs2"

    nt = 100
    ny = 2
    nx = 3

    shape = nt, ny, nx

    d = np.random.random(shape)
    d[10, :, :] = np.nan
    d[11, :, :] = 0
    d[12, :, :] = 1e-10
    d[13, :, :] = 1e10

    # >>> from pyproj import Proj
    # >>> utm = Proj(32633)
    # >>> utm(12.0, 55.0)
    # east = 308124
    # north = 6098907

    x0 = 308124
    y0 = 6098907

    grid = Grid2D(nx=nx, ny=ny, x0=x0, y0=y0, dx=100, dy=100, projection="UTM-33")
    da = mikeio.DataArray(
        data=d, time=pd.date_range("2012-1-1", freq="s", periods=100), geometry=grid
    )
    da.to_dfs(fp)

    ds = mikeio.read(fp)
    assert ds.geometry.dx == 100
    assert ds.geometry.dy == 100
    # shifted x0y0 to origin as not provided in construction of Grid2D
    assert ds.geometry._x0 == 0.0
    assert ds.geometry._y0 == 0.0
    assert ds.geometry.origin[0] == pytest.approx(x0)
    assert ds.geometry.origin[1] == pytest.approx(y0)

    grid = Grid2D(
        nx=nx,
        ny=ny,
        x0=x0,
        y0=y0,
        dx=100,
        dy=100,
        projection="UTM-33",
        # origin=(0.0, 0.0),
    )
    da = mikeio.DataArray(
        data=d, time=pd.date_range("2012-1-1", freq="s", periods=100), geometry=grid
    )
    da.to_dfs(fp)

    ds2 = mikeio.read(fp)
    assert ds2.geometry.dx == 100
    assert ds2.geometry.dy == 100
    # CHANGED: NOT shifted x0y0 to origin as origin was explicitly set to (0,0)
    # assert ds2.geometry._x0 == pytest.approx(x0)
    # assert ds2.geometry._y0 == pytest.approx(y0)
    assert ds2.geometry.origin[0] == pytest.approx(x0)
    assert ds2.geometry.origin[1] == pytest.approx(y0)

    grid = Grid2D(nx=nx, ny=ny, origin=(x0, y0), dx=100, dy=100, projection="UTM-33")
    da = mikeio.DataArray(
        data=d, time=pd.date_range("2012-1-1", freq="s", periods=100), geometry=grid
    )
    da.to_dfs(fp)

    ds3 = mikeio.read(fp)
    assert ds3.geometry.dx == 100
    assert ds3.geometry.dy == 100
    # shifted x0y0 to origin as not provided in construction of Grid2D
    assert ds3.geometry._x0 == 0.0
    assert ds3.geometry._y0 == 0.0
    assert ds3.geometry.origin[0] == pytest.approx(x0)
    assert ds3.geometry.origin[1] == pytest.approx(y0)


def test_write_without_time(tmp_path):
    fp = tmp_path / "utm.dfs2"

    ny = 2
    nx = 3
    grid = Grid2D(nx=nx, ny=ny, dx=100, dy=100, projection="UTM-33")

    d = np.random.random((ny, nx))
    time = pd.date_range("2012-1-1", freq="s", periods=1)
    da = mikeio.DataArray(data=d, time=time, geometry=grid)
    da.to_dfs(fp)

    ds = mikeio.read(fp)
    assert ds.geometry.ny == ny
    assert ds.geometry.nx == nx
    assert ds.shape == (1, ny, nx)

    ds = mikeio.read(fp, time=0)
    assert ds.geometry.ny == ny
    assert ds.geometry.nx == nx
    assert ds.shape == (ny, nx)


def test_read(dfs2_random):

    dfs = dfs2_random
    assert isinstance(dfs.geometry, Grid2D)
    ds = dfs.read(items=["testing water level"])
    data = ds[0].to_numpy()
    assert data[0, 88, 0] == 0
    assert np.isnan(data[0, 89, 0])
    assert data.shape == (3, 100, 2)  # time, y, x


def test_read_bad_item(dfs2_random):
    dfs = dfs2_random
    with pytest.raises(ItemsError) as ex:
        dfs.read(items=100)

    assert ex.value.n_items_file == 1


def test_read_temporal_subset_slice():

    filename = r"tests/testdata/eq.dfs2"
    dfs = mikeio.open(filename)
    ds = dfs.read(time=slice("2000-01-01 00:00", "2000-01-01 12:00"))

    assert len(ds.time) == 13


def test_read_area_subset_bad_bbox():

    filename = "tests/testdata/europe_wind_long_lat.dfs2"
    bbox = (10, 40, 20)
    with pytest.raises(ValueError):
        mikeio.read(filename, area=bbox)


def test_read_area_subset_geo():
    # x: [-15, -14.75, ..., 40] (nx=221, dx=0.25)
    # y: [30, 30.25, ..., 55] (ny=101, dy=0.25)
    filename = "tests/testdata/europe_wind_long_lat.dfs2"
    bbox = (10, 40, 20, 50)
    dsall = mikeio.read(filename)
    dssel = dsall.sel(area=bbox)  # selects all pixels with element center in the bbox
    ds = mikeio.read(filename, area=bbox)
    assert ds.geometry == dssel.geometry
    assert ds.geometry.bbox.left < bbox[0]  #
    assert ds.geometry.x[0] == pytest.approx(bbox[0])
    assert ds.geometry.x[-1] == pytest.approx(bbox[2])
    assert ds.geometry.y[0] == pytest.approx(bbox[1])
    assert ds.geometry.y[-1] == pytest.approx(bbox[3])


def test_subset_bbox():
    filename = "tests/testdata/europe_wind_long_lat.dfs2"
    ds = mikeio.read(filename)
    dssel = ds.sel(area=ds.geometry.bbox)  # this is the entire area
    assert ds.geometry == dssel.geometry


def test_read_area_subset():

    filename = "tests/testdata/eq.dfs2"
    bbox = [10, 4, 12, 7]

    dsall = mikeio.read(filename)
    dssel = dsall.sel(area=bbox)

    ds = mikeio.read(filename, area=bbox)
    assert ds.shape == (25, 4, 3)
    assert ds.geometry == dssel.geometry

    g = ds.geometry
    assert g.ny == 4
    assert g.nx == 3
    assert isinstance(g, Grid2D)

    ds1 = mikeio.read(filename)
    ds2 = ds1.sel(area=bbox)
    assert ds2.shape == (25, 4, 3)
    assert ds2[0].values[4, 2, 1] == ds[0].values[4, 2, 1]

    da2 = ds1[0].sel(area=bbox)
    assert da2.shape == (25, 4, 3)
    assert da2.values[4, 2, 1] == ds[0].values[4, 2, 1]


def test_read_numbered_access(dfs2_random_2items):

    dfs = dfs2_random_2items

    res = dfs.read(items=[1])

    assert np.isnan(res[0].to_numpy()[0, 0, 0])
    assert res.time is not None
    assert res.items[0].name == "Untitled"


def test_properties_vertical_nonutm(dfs2_vertical_nonutm):
    dfs = dfs2_vertical_nonutm
    assert dfs.x0 == 0
    assert dfs.y0 == 0
    assert dfs.dx == pytest.approx(0.01930449)
    assert dfs.dy == 1
    assert dfs.nx == 41
    assert dfs.ny == 76
    assert dfs.longitude == 0
    assert dfs.latitude == 0
    assert dfs.orientation == 0
    assert dfs.n_items == 4
    assert dfs.n_timesteps == 13

    g = dfs.geometry
    assert g.x[0] == dfs.x0
    assert g.y[0] == dfs.y0
    assert g.dx == dfs.dx
    assert g.dy == 1
    assert g.orientation == 0


def test_isel_vertical_nonutm(dfs2_vertical_nonutm):
    ds = dfs2_vertical_nonutm.read()
    assert ds.geometry.is_vertical
    dssel = ds.isel(y=slice(45, None))
    g = dssel.geometry
    assert g.is_vertical
    assert g._x0 == 0
    assert g._y0 == 0  # TODO: should this be 45?
    assert g.x[0] == 0
    assert g.y[0] == 45
    assert g.origin[0] == 0
    assert g.origin[1] == 45  # TODO: should this be 0?


def test_properties_pt_spectrum(dfs2_pt_spectrum):
    dfs = dfs2_pt_spectrum
    assert dfs.x0 == pytest.approx(0.055)
    assert dfs.y0 == 0
    assert dfs.dx == pytest.approx(1.1)
    assert dfs.dy == 22.5
    assert dfs.nx == 25
    assert dfs.ny == 16
    assert dfs.longitude == 0
    assert dfs.latitude == 0
    assert dfs.orientation == 0
    assert dfs.n_items == 1
    assert dfs.n_timesteps == 31

    g = dfs.geometry
    assert g.is_spectral
    assert g.x[0] == pytest.approx(0.055)
    # assert g.x[-1] > 25  # if considered linear
    assert g.x[-1] < 0.6  # logarithmic
    assert g.y[0] == 0
    assert g.dx == pytest.approx(1.1)
    assert g.dy == 22.5
    assert g.orientation == 0


def test_properties_pt_spectrum_linearf(dfs2_pt_spectrum_linearf):
    dfs = dfs2_pt_spectrum_linearf
    # This file doesn't have a valid projection string
    # dfs.FileInfo.Projection.WKTString = ''

    assert dfs.x0 == pytest.approx(0.00390625)
    assert dfs.y0 == 0
    assert dfs.dx == pytest.approx(0.00390625)
    assert dfs.dy == 10
    assert dfs.nx == 128
    assert dfs.ny == 37
    assert dfs.longitude == 0
    assert dfs.latitude == 0
    assert dfs.orientation == 0
    assert dfs.n_items == 1
    assert dfs.n_timesteps == 1

    g = dfs.geometry
    assert g.x[0] == pytest.approx(0.00390625)
    assert g.x[-1] == 0.5  # linear
    assert g.y[0] == 0
    assert g.dx == pytest.approx(0.00390625)
    assert g.dy == 10
    assert g.orientation == 0

    g.is_spectral = True
    assert g.x[-1] == 0.5  # still linear


def test_dir_wave_spectra_relative_time_axis():
    ds = mikeio.open(
        "tests/testdata/dir_wave_analysis_spectra.dfs2", type="spectral"
    ).read()
    assert ds.n_items == 1
    assert ds.geometry.nx == 128
    assert ds.geometry.ny == 37
    assert ds.n_timesteps == 1
    da = ds["Directional spectrum [1]"]
    assert da.type == EUMType._3D_Surface_Elevation_Spectrum


def test_properties_rotated_longlat():
    filepath = Path("tests/testdata/gebco_sound_crop_rotate.dfs2")
    with pytest.raises(ValueError, match="Orientation is not supported for LONG/LAT"):
        mikeio.open(filepath)


def test_properties_rotated_UTM():
    filepath = Path("tests/testdata/BW_Ronne_Layout1998_rotated.dfs2")
    dfs = mikeio.open(filepath)
    g = dfs.geometry
    assert dfs.x0 == 0
    assert dfs.y0 == 0
    assert dfs.dx == 5
    assert dfs.dy == 5
    assert dfs.nx == 263
    assert dfs.ny == 172
    assert dfs.longitude == pytest.approx(14.6814730403)
    assert dfs.latitude == pytest.approx(55.090063)
    assert dfs.orientation == pytest.approx(-22.2387902)
    assert g.orientation == dfs.orientation
    # origin is projected coordinates
    assert g.origin == pytest.approx((479670, 6104860))


def test_select_area_rotated_UTM(tmp_path):
    filepath = Path("tests/testdata/BW_Ronne_Layout1998_rotated.dfs2")
    ds = mikeio.read(filepath)
    assert ds.geometry.origin == pytest.approx((479670, 6104860))
    assert ds.geometry.orientation == pytest.approx(-22.2387902)

    dssel = ds.isel(x=range(10, 20), y=range(15, 45))
    assert dssel.geometry.orientation == ds.geometry.orientation
    assert dssel.geometry.origin == pytest.approx((479673.579, 6104877.669))

    tmpfile = tmp_path / "subset_rotated.dfs2"
    dssel.to_dfs(tmpfile)
    dfs = mikeio.open(tmpfile)
    g = dfs.geometry

    assert dfs.x0 == 0
    assert dfs.y0 == 0
    assert dfs.dx == 5
    assert dfs.dy == 5
    assert dfs.nx == 10
    assert dfs.ny == 30
    assert dfs.orientation == pytest.approx(ds.geometry.orientation)
    assert g.orientation == dfs.orientation
    # origin is in projected coordinates
    assert g.origin == pytest.approx(dssel.geometry.origin)


def test_select_area_rotated_UTM_2():
    fn = Path("tests/testdata/BW_Ronne_Layout1998_rotated.dfs2")
    ds = mikeio.read(fn)
    dssel = ds.isel(x=range(50, 61), y=range(75, 106))
    g1 = dssel.geometry

    # compare to file that has been cropped in MIKE Zero
    fn = Path("tests/testdata/BW_Ronne_Layout1998_rotated_crop.dfs2")
    ds2 = mikeio.read(fn)
    g2 = ds2.geometry

    assert g1.origin == pytest.approx(g2.origin)
    assert g1.dx == g2.dx
    assert g1.dy == g2.dy
    assert g1.nx == g2.nx
    assert g1.ny == g2.ny
    assert g1.orientation == g2.orientation  # orientation in projected coordinates
    assert g1.projection_string == g2.projection_string


def test_write_selected_item_to_new_file(dfs2_random_2items, tmp_path):

    dfs = dfs2_random_2items

    fp = tmp_path / "simple.dfs2"

    ds = dfs.read(items=["Untitled"])

    ds.to_dfs(fp)

    dfs2 = mikeio.open(fp)

    ds2 = dfs2.read()

    assert len(ds2) == 1
    assert ds.items[0].name == "Untitled"
    assert dfs.start_time == dfs2.start_time
    assert dfs.end_time == dfs2.end_time
    assert dfs.projection_string == dfs2.projection_string
    assert dfs.longitude == pytest.approx(dfs2.longitude)
    assert dfs.latitude == pytest.approx(dfs2.latitude)
    assert dfs.orientation == pytest.approx(dfs2.orientation)


def test_repr(dfs2_gebco):

    text = repr(dfs2_gebco)

    assert "Dfs2" in text
    assert "items" in text
    # assert "dx" in text


def test_repr_time(dfs2_random):

    dfs = dfs2_random
    text = repr(dfs)

    assert "Dfs2" in text
    assert "items" in text
    # assert "dx" in text
    assert "steps" in text


def test_write_modified_data_to_new_file(dfs2_gebco, tmp_path):

    dfs = dfs2_gebco

    fp = tmp_path / "mod.dfs2"

    ds = dfs.read()

    ds[0] = ds[0] + 10.0

    ds.to_dfs(fp)

    dfsmod = mikeio.open(fp)

    assert dfs._longitude == dfsmod._longitude


def test_read_some_time_step(dfs2_random_2items):

    dfs = dfs2_random_2items
    res = dfs.read(time=[1, 2])

    assert res[0].to_numpy().shape[0] == 2
    assert len(res.time) == 2


def test_interpolate_non_equidistant_data(tmp_path):

    ds = mikeio.read(
        "tests/testdata/eq.dfs2", time=[0, 2, 3, 6]
    )  # non-equidistant dataset

    assert not ds.is_equidistant

    ds2 = ds.interp_time(dt=3600)

    assert ds2.is_equidistant

    fp = tmp_path / "interpolated_time.dfs2"

    ds2.to_dfs(fp)

    dfs2 = mikeio.open(fp)
    assert dfs2.timestep == 3600.0

    ds3 = dfs2.read()

    assert ds3.is_equidistant


def test_write_some_time_step(tmp_path):

    ds = mikeio.read("tests/testdata/waves.dfs2", time=[1, 2])

    assert ds[0].to_numpy().shape[0] == 2
    assert len(ds.time) == 2

    fp = tmp_path / "waves_subset.dfs2"

    ds.to_dfs(fp)

    dfs2 = mikeio.open(fp)
    assert dfs2.timestep == 86400.0
    assert dfs2.start_time.day == 2


def test_find_by_x_y():
    ds = mikeio.read("tests/testdata/gebco_sound.dfs2")

    ds_point = ds.sel(x=12.74792, y=55.865)
    assert ds_point[0].values[0] == pytest.approx(-43.0)
    assert isinstance(ds_point.geometry, GeometryPoint2D)

    da = ds.Elevation
    da_point = da.sel(x=12.74792, y=55.865)
    assert da_point.values[0] == pytest.approx(-43.0)
    assert isinstance(da_point.geometry, GeometryPoint2D)
    assert da_point.geometry.x == ds_point.geometry.x

    # # with pytest.raises(NotImplementedError):
    # da.sel(x=xx, y=yy)


def test_interp_to_x_y():
    ds = mikeio.read("tests/testdata/gebco_sound.dfs2")

    x = 12.74792
    y = 55.865
    dai = ds.Elevation.interp(x=x, y=y)
    assert dai.values[0] == pytest.approx(-42.69764538978391)

    assert dai.geometry.x == x
    assert dai.geometry.y == y


def test_write_accumulated_datatype(tmp_path):
    fp = tmp_path / "simple.dfs2"

    d = np.random.random([100, 2, 3])

    da = mikeio.DataArray(
        data=d,
        time=pd.date_range("2021-1-1", periods=100, freq="s"),
        geometry=mikeio.Grid2D(nx=3, ny=2, dx=1, dy=1),
        item=ItemInfo(
            "testing water level",
            EUMType.Water_Level,
            EUMUnit.meter,
            data_value_type="MeanStepBackward",
        ),
    )

    da.to_dfs(fp)

    newdfs = mikeio.open(fp)
    assert newdfs.items[0].data_value_type == 3


def test_write_NonEqCalendarAxis(tmp_path):

    fp = tmp_path / "simple.dfs2"

    d = np.random.random([6, 5, 10])
    d[1, :, :] = np.nan
    d[2, :, :] = 0
    d[3, 3:, :] = 2
    d[4, :, 4:] = 5
    da = mikeio.DataArray(
        data=d,
        geometry=mikeio.Grid2D(nx=10, ny=5, dx=100, dy=200),
        time=[
            datetime.datetime(2012, 1, 1),
            datetime.datetime(2012, 1, 4),
            datetime.datetime(2012, 1, 5),
            datetime.datetime(2012, 1, 10),
            datetime.datetime(2012, 1, 15),
            datetime.datetime(2012, 1, 28),
        ],
    )

    da.to_dfs(fp)

    newds = mikeio.read(fp)
    assert newds.is_equidistant is False
    assert newds.start_time.year == 2012
    assert newds.end_time.day == 28


def test_write_non_equidistant_data(tmp_path):

    ds = mikeio.read(
        "tests/testdata/eq.dfs2", time=[0, 2, 3, 6]
    )  # non-equidistant dataset

    assert not ds.is_equidistant

    fp = tmp_path / "neq_from_dataset.dfs2"

    ds.to_dfs(fp)

    ds3 = mikeio.read(fp)

    assert not ds3.is_equidistant


def test_read_concat_write_dfs2(tmp_path):
    outfilename = tmp_path / "waves_concat.dfs2"

    ds1 = mikeio.read("tests/testdata/waves.dfs2", time=[0, 1])
    # ds2 = mikeio.read("tests/testdata/waves.dfs2", time=2)  # dont do this, it will not work, since reading a single time step removes the time dimension
    ds2 = mikeio.read("tests/testdata/waves.dfs2", time=[2], keepdims=True)
    dsc = mikeio.Dataset.concat([ds1, ds2])
    assert dsc.n_timesteps == 3
    assert dsc.end_time == ds2.end_time
    assert isinstance(dsc.geometry, Grid2D)
    dsc.to_dfs(outfilename)

    dsnew = mikeio.read(outfilename)
    assert isinstance(dsnew.geometry, Grid2D)
    assert dsnew.n_timesteps == 3
    assert dsnew.end_time == ds2.end_time


def test_spatial_aggregation_dfs2_to_dfs0(tmp_path):

    outfilename = tmp_path / "waves_max.dfs0"

    ds = mikeio.read("tests/testdata/waves.dfs2")
    ds_max = ds.nanmax(axis="space")
    ds_max.to_dfs(outfilename)

    dsnew = mikeio.read(outfilename)

    assert dsnew.n_timesteps == ds.n_timesteps
    assert dsnew.n_items == ds.n_items


def test_da_to_xarray():
    ds = mikeio.read("tests/testdata/waves.dfs2")
    da = ds[0]
    xr_da = da.to_xarray()
    assert isinstance(xr_da, xarray.DataArray)


def test_ds_to_xarray():
    ds = mikeio.read("tests/testdata/waves.dfs2")
    xr_ds = ds.to_xarray()
    assert isinstance(xr_ds, xarray.Dataset)


def test_da_plot():
    ds = mikeio.read("tests/testdata/gebco_sound.dfs2")
    da = ds[0]
    da.plot()
    da.plot.contour(title="contour plot")
    da.plot.contourf()
    da.plot.hist(title="histogram plot", alpha=0.2)

    plt.close("all")


def test_grid2d_plot():
    ds = mikeio.read("tests/testdata/gebco_sound.dfs2")
    g = ds[0].geometry
    g.plot(color="0.2", linewidth=2, title="grid plot")
    g.plot.outline(title="outline plot")

    plt.close("all")


def test_read_single_precision():

    ds = mikeio.read("tests/testdata/random.dfs2", items=0, dtype=np.float32)

    assert len(ds) == 1
    assert ds[0].dtype == np.float32


def dfs2_props_to_list(d):

    lon = d._dfs.FileInfo.Projection.Longitude
    lat = d._dfs.FileInfo.Projection.Latitude
    rot = d._dfs.FileInfo.Projection.Orientation
    res = [
        d.x0,
        d.y0,
        d.dx,
        d.dy,
        d.nx,
        d.ny,
        d._projstr,
        lon,
        lat,
        rot,
        d._n_timesteps,
        d._start_time,
        d._dfs.FileInfo.TimeAxis.TimeAxisType,
        d.n_items,
        # d._deletevalue,
    ]

    for item in d.items:
        res.append(item.type)
        res.append(item.unit)
        res.append(item.name)

    return res


def is_header_unchanged_on_read_write(tmp_path, filename):
    dfsA = mikeio.open("tests/testdata/" + filename)
    props_A = dfs2_props_to_list(dfsA)

    ds = dfsA.read()
    filename_out = tmp_path / filename
    ds.to_dfs(filename_out)
    dfsB = mikeio.open(filename_out)
    props_B = dfs2_props_to_list(dfsB)
    for pA, pB in zip(props_A, props_B):
        assert pytest.approx(pA) == pB


def test_read_write_header_unchanged_utm_not_rotated(tmp_path):
    is_header_unchanged_on_read_write(tmp_path, "utm_not_rotated_neurope_temp.dfs2")


def test_read_write_header_unchanged_longlat(tmp_path):
    is_header_unchanged_on_read_write(tmp_path, "europe_wind_long_lat.dfs2")


def test_read_write_header_unchanged_global_longlat(tmp_path):
    is_header_unchanged_on_read_write(
        tmp_path, "global_long_lat_pacific_view_temperature_delta.dfs2"
    )


def test_read_write_header_unchanged_local_coordinates(tmp_path):
    is_header_unchanged_on_read_write(tmp_path, "M3WFM_sponge_local_coordinates.dfs2")


def test_read_write_header_unchanged_utm_rotated(tmp_path):
    is_header_unchanged_on_read_write(tmp_path, "BW_Ronne_Layout1998_rotated.dfs2")


def test_read_write_header_unchanged_vertical(tmp_path):
    is_header_unchanged_on_read_write(tmp_path, "hd_vertical_slice.dfs2")


# def test_read_write_header_unchanged_spectral(tmp_path):
#     # fails: <TimeAxisType.TimeEquidistant: 1> != <TimeAxisType.CalendarEquidistant: 3>
#     is_header_unchanged_on_read_write(tmp_path, "dir_wave_analysis_spectra.dfs2")


def test_read_write_header_unchanged_spectral_2(tmp_path):
    is_header_unchanged_on_read_write(tmp_path, "pt_spectra.dfs2")


def test_read_write_header_unchanged_MIKE_SHE_output(tmp_path):
    is_header_unchanged_on_read_write(tmp_path, "Karup_MIKE_SHE_output.dfs2")


def test_MIKE_SHE_output():
    ds = mikeio.read("tests/testdata/Karup_MIKE_SHE_output.dfs2")
    assert ds.n_timesteps == 6
    assert ds.n_items == 2
    g = ds.geometry
    assert g.x[0] == 494329.0
    assert g.y[0] == pytest.approx(6220250.0)
    assert g.origin == pytest.approx((494329.0, 6220250.0))

    ds2 = ds.isel(x=range(30, 45), y=range(35, 42))
    g2 = ds2.geometry
    assert g2.x[0] == g.x[0] + 30 * g.dx
    assert g2.y[0] == g.y[0] + 35 * g.dy
    assert g2.origin == pytest.approx((g2.x[0], g2.y[0]))


def test_read_dfs2_static_dt_zero():

    with pytest.warns(UserWarning, match="positive"):
        ds = mikeio.read("tests/testdata/single_time_dt_zero.dfs2")
    assert ds.n_timesteps == 1
    assert ds.shape == (1, 2, 2)

    with pytest.warns(UserWarning, match="positive"):
        ds2 = mikeio.read("tests/testdata/single_time_dt_zero.dfs2", time=0)

    assert ds2.shape == (2, 2)
    assert "time" not in ds2.dims



def test_add_custom_block_to_dfs2(tmp_path):
    ds = mikeio.read("tests/testdata/gebco_sound.dfs2")
    fp = tmp_path / "gebco_sound_with_custom_block.dfs2"
    ds.to_dfs(
        fp,
        custom_blocks={"M21_Misc": np.array([327, 0.2, -900, 10, 0, 0, 0], np.float32)},
    )

    ds2 = mikeio.open(fp)
    assert ds2._dfs.FileInfo.CustomBlocks[0].Name == "M21_Misc"

def test_write_read_local_coordinates(tmp_path):
    da = mikeio.DataArray(
        np.array([[1, 2, 3], [4, 5, 6]]),
        geometry=mikeio.Grid2D(nx=3, ny=2, dx=0.5, projection="NON-UTM"),
    )
    fp = tmp_path / "local_coordinates.dfs2"
    da.to_dfs(fp)

    ds = mikeio.read(fp)
    assert da.geometry == ds.geometry


def test_to_xarray():
    # data is not important here
    data = np.array([[1, 2, 3], [4, 5, 6]])

    # geographic coordinates
    dag = mikeio.DataArray(
        data=data,
        geometry=mikeio.Grid2D(nx=3, ny=2, dx=0.5, projection="LONG/LAT"),
    )
    assert dag.geometry.x[0] == pytest.approx(0.0)
    assert dag.geometry.y[0] == pytest.approx(0.0)
    xr_dag = dag.to_xarray()
    assert xr_dag.x[0] == pytest.approx(0.0)
    assert xr_dag.y[0] == pytest.approx(0.0)

    # local coordinates
    da = mikeio.DataArray(
        data=data,
        geometry=mikeio.Grid2D(nx=3, ny=2, dx=0.5, projection="NON-UTM"),
    )
    # local coordinates (=NON-UTM) have a different convention, geometry.x still refers to element centers
    assert da.geometry.x[0] == pytest.approx(0.25)
    assert da.geometry.y[0] == pytest.approx(0.25)

    xr_da = da.to_xarray()
    assert xr_da.x[0] == pytest.approx(0.25)
    assert xr_da.y[0] == pytest.approx(0.25)
   
