from pathlib import Path
import numpy as np
import pytest
import mikeio
from mikeio import Mesh


@pytest.fixture
def tri_mesh() -> Mesh:
    return Mesh("tests/testdata/odense_rough.mesh")


@pytest.fixture
def mixed_mesh() -> Mesh:
    return Mesh("tests/testdata/quad_tri.mesh")


def test_read_mesh_from_path():
    testdata = Path("tests/testdata")
    fp = testdata / "odense_rough.mesh"
    msh = Mesh(fp)
    assert msh.n_nodes == 399


def test_get_number_of_elements(tri_mesh):
    msh = tri_mesh
    assert msh.n_elements == 654


def test_element_coordinates(tri_mesh):
    msh = tri_mesh

    ec = msh.element_coordinates

    assert ec.shape == (654, 3)
    assert ec[0, 0] > 212000.0
    assert ec[0, 1] > 6153000.0


def test_read_mixed_mesh(mixed_mesh):
    msh = mixed_mesh
    assert msh.n_nodes == 798

    el_tbl_vec = np.hstack(msh.element_table)
    assert len(el_tbl_vec) > 3 * msh.n_elements
    assert len(el_tbl_vec) < 4 * msh.n_elements
    assert np.all(el_tbl_vec >= 0)
    assert np.all(el_tbl_vec < msh.n_nodes)


def test_read_write_mixed_mesh(mixed_mesh, tmp_path):
    msh = mixed_mesh
    outfilename = tmp_path / "quad_tri_v2.mesh"
    msh.write(outfilename)

    msh2 = Mesh(outfilename)

    assert outfilename.exists()

    assert np.all(np.hstack(msh2.element_table) == np.hstack(msh.element_table))
    assert np.all(msh2.element_coordinates == msh.element_coordinates)


def test_node_coordinates(tri_mesh):
    msh = tri_mesh

    nc = msh.node_coordinates

    assert nc.shape == (399, 3)


def test_get_land_node_coordinates(tri_mesh):
    msh = tri_mesh

    nc = msh.node_coordinates[msh.geometry.codes == 1]

    assert nc.shape == (134, 3)


def test_get_bad_node_coordinates(tri_mesh):
    msh = tri_mesh

    with pytest.raises(Exception):
        msh.get_node_coords(code="foo")


def test_set_z(tri_mesh):
    msh = tri_mesh
    nc = msh.node_coordinates.copy()
    assert msh.element_coordinates[:, 2].min() == pytest.approx(-10.938001)
    zn = msh.node_coordinates[:, 2]
    nc[zn < -3, 2] = -3

    # setting the property, triggers update of element coordinates
    msh.node_coordinates = nc

    assert msh.element_coordinates[:, 2].min() == pytest.approx(-3)

    zn = msh.node_coordinates[:, 2]
    assert zn.min() == -3


def test_set_codes(tri_mesh):
    msh = tri_mesh
    codes = msh.geometry.codes
    assert msh.geometry.codes[2] == 2
    codes[codes == 2] = 7  # work directly on reference

    assert msh.geometry.codes[2] == 7

    new_codes = msh.geometry.codes.copy()
    new_codes[new_codes == 7] = 9
    msh.geometry.codes = new_codes  # assign from copy

    assert msh.geometry.codes[2] == 9

    with pytest.raises(ValueError):
        # not same length
        msh.geometry.codes = codes[0:4]


def test_write(tri_mesh, tmp_path):
    outfilename = tmp_path / "simple.mesh"
    msh = tri_mesh

    msh.write(outfilename)

    assert outfilename.exists()


def test_write_part_isel(tri_mesh, tmp_path):
    outfilename = tmp_path / "simple_sub.mesh"

    msh = tri_mesh

    gsub = msh.geometry.isel(range(50, 100))
    gsub.to_mesh(outfilename)

    assert outfilename.exists()


def test_write_mesh_from_dfsu(tmp_path):
    outfilename = tmp_path / "quad_tri.mesh"
    dfsufilename = "tests/testdata/FakeLake.dfsu"

    dfs = mikeio.open(dfsufilename)

    geometry = dfs.geometry

    geometry.to_mesh(outfilename)

    msh2 = Mesh(outfilename)

    assert outfilename.exists()

    assert np.all(np.hstack(msh2.element_table) == np.hstack(geometry.element_table))


def test_to_shapely(tri_mesh) -> None:
    msh = tri_mesh
    shp = msh.to_shapely()
    assert shp.geom_type == "MultiPolygon"
    assert shp.area == pytest.approx(68931409.58160606)
