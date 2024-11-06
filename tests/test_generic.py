import numpy as np
import pandas as pd
import mikeio
from mikeio import generic
from mikeio.generic import scale, diff, sum, extract, avg_time, fill_corrupt
import pytest


def test_add_constant(tmp_path):

    infilename = "tests/testdata/random.dfs0"
    fp = tmp_path / "add.dfs0"
    scale(infilename, fp, offset=100.0)

    org = mikeio.read(infilename)

    scaled = mikeio.read(fp)

    orgvalue = org[0].values[0]
    expected = orgvalue + 100.0
    scaledvalue = scaled[0].values[0]
    assert scaledvalue == pytest.approx(expected)


def test_multiply_constant(tmp_path):

    infilename = "tests/testdata/random.dfs0"
    fp = tmp_path / "mult.dfs0"
    scale(infilename, fp, factor=1.5)

    org = mikeio.read(infilename)

    scaled = mikeio.read(fp)

    orgvalue = org[0].values[0]
    expected = orgvalue * 1.5
    scaledvalue = scaled[0].values[0]
    assert scaledvalue == pytest.approx(expected)


def test_multiply_constant_single_item_number(tmp_path):

    infilename = "tests/testdata/wind_north_sea.dfsu"
    fp = tmp_path / "mult.dfsu"
    scale(infilename, fp, factor=1.5, items=[0])

    org = mikeio.read(infilename)

    scaled = mikeio.read(fp)

    orgvalue_speed = org[0].values[0][0]
    expected_speed = orgvalue_speed * 1.5
    scaledvalue_speed = scaled[0].values[0][0]
    assert scaledvalue_speed == pytest.approx(expected_speed)

    orgvalue_dir = org[1].values[0, 0]
    expected_dir = orgvalue_dir
    scaledvalue_dir = scaled[1].values[0, 0]
    assert scaledvalue_dir == pytest.approx(expected_dir)


def test_multiply_constant_single_item_name(tmp_path):

    infilename = "tests/testdata/wind_north_sea.dfsu"
    fp = tmp_path / "multname.dfsu"
    scale(infilename, fp, factor=1.5, items=["Wind speed"])

    org = mikeio.read(infilename)

    scaled = mikeio.read(fp)

    orgvalue_speed = org["Wind speed"].to_numpy()[0, 0]
    expected_speed = orgvalue_speed * 1.5
    scaledvalue_speed = scaled["Wind speed"].to_numpy()[0, 0]
    assert scaledvalue_speed == pytest.approx(expected_speed)

    orgvalue_dir = org["Wind direction"].to_numpy()[0, 0]
    expected_dir = orgvalue_dir
    scaledvalue_dir = scaled["Wind direction"].to_numpy()[0, 0]
    assert scaledvalue_dir == pytest.approx(expected_dir)


def test_diff_itself(tmp_path):

    infilename_1 = "tests/testdata/gebco_sound.dfs2"
    infilename_2 = "tests/testdata/gebco_sound.dfs2"
    fp = tmp_path / "diff.dfs2"

    diff(infilename_1, infilename_2, fp)

    org = mikeio.read(infilename_1)

    assert np.isnan(org["Elevation"].to_numpy()[0, -1, -1])

    diffed = mikeio.read(fp)

    diffedvalue = diffed["Elevation"].to_numpy()[0, 0, 0]
    assert diffedvalue == pytest.approx(0.0)
    assert np.isnan(diffed["Elevation"].to_numpy()[0, -1, -1])


def test_sum_itself(tmp_path):

    infilename_1 = "tests/testdata/gebco_sound.dfs2"
    infilename_2 = "tests/testdata/gebco_sound.dfs2"
    fp = tmp_path / "diff.dfs2"

    sum(infilename_1, infilename_2, fp)

    mikeio.read(infilename_1)

    summed = mikeio.read(fp)

    assert np.isnan(summed["Elevation"].to_numpy()[0][-1, -1])


def test_add_constant_delete_values_unchanged(tmp_path):

    infilename = "tests/testdata/gebco_sound.dfs2"
    fp = tmp_path / "adj.dfs2"
    scale(infilename, fp, offset=-2.1, items=["Elevation"])

    org = mikeio.read(infilename)

    scaled = mikeio.read(fp)

    orgvalue = org["Elevation"].to_numpy()[0, 0, 0]
    scaledvalue = scaled["Elevation"].to_numpy()[0, 0, 0]
    assert scaledvalue == pytest.approx(orgvalue - 2.1)

    orgvalue = org["Elevation"].to_numpy()[0, 100, 0]
    assert np.isnan(orgvalue)

    scaledvalue = scaled["Elevation"].to_numpy()[0, 100, 0]
    assert np.isnan(scaledvalue)


def test_multiply_constant_delete_values_unchanged_2(tmp_path):

    infilename = "tests/testdata/random_two_item.dfs2"
    fp = tmp_path / "adj.dfs2"

    item_name = "testing water level"

    scale(infilename, fp, factor=1000.0, items=[item_name])

    org = mikeio.read(infilename)

    scaled = mikeio.read(fp)

    orgvalue = org[item_name].to_numpy()[0, 0, 0]
    scaledvalue = scaled[item_name].to_numpy()[0, 0, 0]
    assert scaledvalue == pytest.approx(orgvalue * 1000.0)

    orgvalue = org[item_name].to_numpy()[0, -11, 0]
    assert np.isnan(orgvalue)

    scaledvalue = scaled[item_name].to_numpy()[0, -11, 0]
    assert np.isnan(scaledvalue)


def test_linear_transform(tmp_path):

    infilename = "tests/testdata/random.dfs0"
    fp = tmp_path / "linear.dfs0"
    scale(infilename, fp, offset=-20.0, factor=1.5)

    org = mikeio.read(infilename)

    scaled = mikeio.read(fp)

    orgvalue = org[0].values[0]
    expected = orgvalue * 1.5 - 20.0
    scaledvalue = scaled[0].values[0]
    assert scaledvalue == pytest.approx(expected)


def test_linear_transform_dfsu(tmp_path):

    infilename = "tests/testdata/HD2D.dfsu"
    fp = tmp_path / "linear.dfsu"
    scale(infilename, fp, offset=-20.0, factor=1.5)

    org = mikeio.read(infilename)

    scaled = mikeio.read(fp)

    orgvalue = org[0].values[0]
    expected = orgvalue * 1.5 - 20.0
    scaledvalue = scaled[0].values[0]
    assert scaledvalue == pytest.approx(expected)


def test_sum_dfsu(tmp_path):

    infilename_a = "tests/testdata/HD2D.dfsu"
    infilename_b = "tests/testdata/HD2D.dfsu"
    fp = tmp_path / "sum.dfsu"
    mikeio.generic.sum(infilename_a, infilename_b, fp)

    org = mikeio.read(infilename_a)

    summed = mikeio.read(fp)

    orgvalue = org[0].values[0]
    expected = orgvalue * 2
    scaledvalue = summed[0].values[0]
    assert scaledvalue == pytest.approx(expected)


def test_diff_dfsu(tmp_path):

    infilename_a = "tests/testdata/HD2D.dfsu"
    infilename_b = "tests/testdata/HD2D.dfsu"
    fp = tmp_path / "diff.dfsu"
    mikeio.generic.diff(infilename_a, infilename_b, fp)

    mikeio.read(infilename_a)

    diffed = mikeio.read(fp)

    expected = 0.0
    scaledvalue = diffed[0].values[0]
    assert scaledvalue == pytest.approx(expected)


def test_concat_overlapping(tmp_path):
    infilename_a = "tests/testdata/tide1.dfs1"
    infilename_b = "tests/testdata/tide2.dfs1"
    fp = tmp_path / "concat.dfs1"

    mikeio.generic.concat([infilename_a, infilename_b], fp)

    ds = mikeio.read(fp)
    assert len(ds.time) == 145


def test_concat_average(tmp_path):
    infilename_a = "generic_concat_file1.dfs2"
    infilename_b = "generic_concat_file2.dfs2"
    fp = tmp_path / "concat.dfs1"

    mikeio.generic.concat([infilename_a, infilename_b], fp, keep="average")

    ds = mikeio.read(fp)
    assert ds[0][2].values[0][0] == 51.5


def test_concat_files_gap_fail(tmp_path):
    infilename_a = "tests/testdata/tide1.dfs1"
    infilename_b = "tests/testdata/tide4.dfs1"
    fp = tmp_path / "concat.dfs1"
    with pytest.raises(Exception):
        mikeio.generic.concat([infilename_a, infilename_b], fp)


def test_concat_three_files(tmp_path):
    infiles = [
        "tests/testdata/tide1.dfs1",
        "tests/testdata/tide2.dfs1",
        "tests/testdata/tide4.dfs1",
    ]
    fp = tmp_path / "concat.dfs1"

    mikeio.generic.concat(infiles, fp)

    ds = mikeio.read(fp)
    assert len(ds.time) == (5 * 48 + 1)


def test_concat_keep(tmp_path):
    """
    test keep arguments of concatenation function
    """
    # added keep arguments to test
    keep_args = ["first", "last"]

    infiles = [
        "tests/testdata/tide1.dfs1",
        "tests/testdata/tide2_offset.dfs1",
        "tests/testdata/tide4.dfs1",
    ]
    fp = tmp_path / "concat.dfs1"

    # loop through keep args to test all
    for keep_arg in keep_args:
        mikeio.generic.concat(infilenames=infiles, outfilename=fp, keep=keep_arg)

        # read outfile
        dso = mikeio.read(fp)
        df_o = pd.DataFrame(dso[0].to_numpy(), index=dso.time)

        """ handle data (always keep two data in memory to compare in overlapping period) """
        # !!! checks below solely consider one item !!
        for i, infile in enumerate(infiles):
            dsi = mikeio.read(infile)

            # just store current (=last) data in first loop. no check required
            if i == 0:
                df_last = pd.DataFrame(dsi[0].to_numpy(), index=dsi.time)
            else:
                # move previously last to first
                df_first = df_last.copy()
                # overwrite last with new / current data
                df_last = pd.DataFrame(dsi[0].to_numpy(), index=dsi.time)

                # find overlapping timesteps via merge
                overlap_idx = pd.merge(
                    df_first, df_last, how="inner", left_index=True, right_index=True
                ).index

                # check if first and output are identical (results in dataframe with overlapping datetime index and bool values)
                first_out = (
                    (df_first.loc[overlap_idx] == df_o.loc[overlap_idx])
                    .eq(True)
                    .all()
                    .all()
                )
                last_out = (
                    (df_last.loc[overlap_idx] == df_o.loc[overlap_idx])
                    .eq(True)
                    .all()
                    .all()
                )

                if keep_arg == "first":
                    assert first_out, "overlap should be with first dataset"
                elif keep_arg == "last":
                    assert last_out, "overlap should be with last dataset"


def test_concat_non_equidistant_dfs0(tmp_path):

    # create two non-equidistant dfs0 files
    da1 = mikeio.DataArray(
        data=np.array([0.0, 0.1, 0.2]),
        time=pd.DatetimeIndex(
            ["2017-10-27 01:58", "2017-10-27 04:32", "2017-10-27 05:32"]
        ),
    )
    assert not da1.is_equidistant
    da2 = mikeio.DataArray(
        data=np.array([0.9, 1.1, 0.2]),
        time=pd.DatetimeIndex(
            ["2017-10-28 01:59", "2017-10-29 04:32", "2017-11-01 05:32"]
        ),
    )
    assert not da2.is_equidistant
    files = [tmp_path / "da1.dfs0", tmp_path / "da2.dfs0"]
    da1.to_dfs(tmp_path / "da1.dfs0")
    da2.to_dfs(tmp_path / "da2.dfs0")

    # concatenate
    fp = tmp_path / "concat.dfs0"
    mikeio.generic.concat(files, fp)

    ds = mikeio.read(fp)
    assert len(ds.time) == 6

    assert ds.sel(time="2017-10-29 04:32").to_dataframe().iloc[0, 0] == pytest.approx(
        1.1
    )


def test_extract_equidistant(tmp_path):

    infile = "tests/testdata/waves.dfs2"
    fp = tmp_path / "waves_subset.dfs2"

    extract(infile, fp, start=1, end=-1)
    orig = mikeio.read(infile)
    extracted = mikeio.read(fp)
    assert extracted.n_timesteps == orig.n_timesteps - 1
    assert extracted.time[0] == orig.time[1]

    with pytest.raises(ValueError):
        extract(infile, fp, start=-23, end=1000)

    with pytest.raises(ValueError):
        extract(infile, fp, start=-23.5)

    with pytest.raises(ValueError):
        extract(infile, fp, start=1000)


def test_extract_non_equidistant(tmp_path):

    infile = "tests/testdata/da_diagnostic.dfs0"
    fp = tmp_path / "da_diagnostic_subset.dfs0"

    extract(infile, fp, start="2017-10-27 01:58", end="2017-10-27 04:32")
    extracted = mikeio.read(fp)
    assert extracted.n_timesteps == 16
    assert extracted.time[0].hour == 2

    extract(infile, fp, end=3600.0)
    extracted = mikeio.read(fp)
    assert extracted.n_timesteps == 7
    assert extracted.time[0].hour == 0
    assert extracted.time[-1].minute == 0

    with pytest.raises(ValueError):
        extract(infile, fp, start=1800.0, end=237981231.232)

    extract(infile, fp, "2017-10-27,2017-10-28")
    extracted = mikeio.read(fp)
    assert extracted.n_timesteps > 0

    with pytest.raises(ValueError):
        extract(infile, fp, start=7200.0, end=1800.0)


def test_extract_relative_time_axis(tmp_path):

    infile = "tests/testdata/eq_relative.dfs0"
    fp = tmp_path / "eq_relative_subset.dfs0"

    with pytest.raises(Exception):
        extract(infile, fp, start=0, end=4)


def test_extract_step_equidistant(tmp_path):

    infile = "tests/testdata/tide1.dfs1"  # 30min
    fp = tmp_path / "tide1_step12.dfs1"

    extract(infile, fp, step=12)
    orig = mikeio.read(infile)
    extracted = mikeio.read(fp)
    assert extracted.n_timesteps == 9
    assert extracted.timestep == 6 * 3600
    assert extracted.time[0] == orig.time[0]
    assert extracted.time[-1] == orig.time[-1]


def test_extract_step_non_equidistant(tmp_path):

    infile = "tests/testdata/da_diagnostic.dfs0"
    fp = tmp_path / "da_diagnostic_step3.dfs0"

    extract(infile, fp, step=3)
    orig = mikeio.read(infile)
    extracted = mikeio.read(fp)
    assert extracted.n_timesteps * 3 == orig.n_timesteps
    assert extracted.time[0] == orig.time[0]
    assert extracted.time[-1] == orig.time[-3]


def test_extract_items(tmp_path):

    # This is a Dfsu 3d file (i.e. first item is Z coordinate)
    infile = "tests/testdata/oresund_vertical_slice.dfsu"
    fp = tmp_path / "oresund_vertical_slice_extract.dfsu"

    extract(infile, fp, items="Temperature")
    extracted = mikeio.read(fp)
    assert extracted.n_items == 1
    assert extracted.items[0].name == "Temperature"

    extract(infile, fp, items=[1])
    extracted = mikeio.read(fp)
    assert extracted.n_items == 1
    assert extracted.items[0].name == "Salinity"

    extract(infile, fp, items=range(0, 2))  # [0,1]
    extracted = mikeio.read(fp)
    assert extracted.n_items == 2
    assert extracted.items[0].name == "Temperature"

    extract(infile, fp, items=["Salinity", 0])
    extracted = mikeio.read(fp)
    assert extracted.n_items == 2

    with pytest.raises(Exception):
        # must be unique
        extract(infile, fp, items=["Salinity", 1])

    with pytest.raises(Exception):
        # no negative numbers
        extract(infile, fp, items=[0, 2, -1])

    with pytest.raises(Exception):
        extract(infile, fp, items=[0, "not_an_item"])


def test_time_average(tmp_path):

    infilename = "tests/testdata/NorthSea_HD_and_windspeed.dfsu"
    fp = tmp_path / "NorthSea_HD_and_windspeed_avg.dfsu"
    avg_time(infilename, fp)

    org = mikeio.read(infilename)

    averaged = mikeio.read(fp)

    assert all([a == b for a, b in zip(org.items, averaged.items)])
    assert org.time[0] == averaged.time[0]
    assert org.shape[1] == averaged.shape[1]
    assert averaged.shape[0] == 1
    assert np.allclose(org.mean(axis=0)[0].to_numpy(), averaged[0].to_numpy())


def test_time_average_dfsu_3d(tmp_path):
    infilename = "tests/testdata/oresund_sigma_z.dfsu"
    fp = tmp_path / "oresund_sigma_z_avg.dfsu"
    avg_time(infilename, fp)

    org = mikeio.open(infilename)
    averaged = mikeio.open(fp)

    assert averaged.n_timesteps == 1
    assert org.start_time == averaged.start_time
    assert org.n_items == averaged.n_items


def test_time_average_deletevalues(tmp_path):

    infilename = "tests/testdata/gebco_sound.dfs2"
    fp = tmp_path / "gebco_sound_avg.dfs2"
    avg_time(infilename, fp)

    org = mikeio.read(infilename)
    averaged = mikeio.read(fp)

    assert all([a == b for a, b in zip(org.items, averaged.items)])
    assert org.time[0] == averaged.time[0]
    assert org.shape[1] == averaged.shape[1]
    nan1 = np.isnan(org[0].to_numpy())
    nan2 = np.isnan(averaged[0].to_numpy())
    assert np.all(nan1 == nan2)
    assert np.allclose(org[0].to_numpy()[~nan1], averaged[0].to_numpy()[~nan2])


def test_quantile_dfsu(tmp_path):

    infilename = "tests/testdata/oresundHD_run1.dfsu"
    fp = tmp_path / "oresund_q10.dfsu"
    generic.quantile(infilename, fp, q=0.1, items=["Surface elevation"])

    org = mikeio.read(infilename).quantile(q=0.1, axis=0)
    q10 = mikeio.read(fp)

    assert np.allclose(org[0].to_numpy(), q10[0].to_numpy())


def test_quantile_dfsu_buffer_size(tmp_path):

    infilename = "tests/testdata/oresundHD_run1.dfsu"
    fp = tmp_path / "oresund_q10.dfsu"
    generic.quantile(infilename, fp, q=0.1, buffer_size=1e5, items=0)

    org = mikeio.read(infilename).quantile(q=0.1, axis=0)
    q10 = mikeio.read(fp)

    assert np.allclose(org[0].to_numpy(), q10[0].to_numpy())


def test_quantile_dfs2(tmp_path):

    infilename = "tests/testdata/eq.dfs2"
    fp = tmp_path / "eq_q90.dfs2"
    generic.quantile(infilename, fp, q=0.9)

    org = mikeio.read(infilename).quantile(q=0.9, axis=0)
    q90 = mikeio.read(fp)

    assert np.allclose(org[0].to_numpy(), q90[0].to_numpy())


def test_quantile_dfs0(tmp_path):

    infilename = "tests/testdata/da_diagnostic.dfs0"
    fp = tmp_path / "da_q001_q05.dfs0"
    generic.quantile(infilename, fp, q=[0.01, 0.5])

    org = mikeio.read(infilename).quantile(q=[0.01, 0.5], axis=0)
    qnt = mikeio.read(fp)

    assert np.allclose(org[0].to_numpy(), qnt[0].to_numpy())
    # assert np.allclose(org[5], qnt[5])


def test_quantile_dfsu_3d(tmp_path):
    infilename = "tests/testdata/oresund_sigma_z.dfsu"
    fp = tmp_path / "oresund_sigma_z_q10_90.dfsu"
    generic.quantile(infilename, fp, q=[0.1, 0.9], items=["Temperature"])

    qd = mikeio.open(fp)
    assert qd.n_timesteps == 1


def test_dfs_ext_capitalisation(tmp_path):
    filename = "tests/testdata/waves2.DFS0"
    ds = mikeio.open(filename)
    ds = mikeio.read(filename)
    ds.to_dfs(tmp_path / "void.DFS0")
    filename = "tests/testdata/oresund_vertical_slice2.DFSU"
    ds = mikeio.open(filename)
    filename = "tests/testdata/odense_rough2.MESH"
    ds = mikeio.open(filename)
    assert True


def test_fill_corrupt_data(tmp_path):
    """This test doesn't verify much..."""

    infile = "tests/testdata/waves.dfs2"

    fp = tmp_path / "waves_filled.dfs2"

    fill_corrupt(infilename=infile, outfilename=fp, items=[1])
    orig = mikeio.read(infile)
    extracted = mikeio.read(fp)
    assert extracted.n_timesteps == orig.n_timesteps
