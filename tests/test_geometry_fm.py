import pytest
import mikeio
from mikeio.spatial import GeometryFM2D, GeometryFM3D
from mikeio.exceptions import OutsideModelDomainError
from mikeio.spatial import GeometryPoint2D


@pytest.fixture
def simple_3d_geom() -> GeometryFM3D:
    #     x     y    z
    nc = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 0.0, -1.0),
        (1.0, 0.0, -1.0),
        (1.0, 1.0, -1.0),
        (0.0, 0.0, -2.0),
        (1.0, 0.0, -2.0),
        (1.0, 1.0, -2.0),
    ]

    el = [(0, 1, 2, 3, 4, 5), (3, 4, 5, 6, 7, 8)]

    g = GeometryFM3D(
        node_coordinates=nc,
        element_table=el,
        projection="LONG/LAT",
        n_layers=2,
        n_sigma=2,
    )

    return g


def test_isel_list_of_indices(simple_3d_geom: GeometryFM3D) -> None:
    g = simple_3d_geom

    g1 = g.isel([0, 1])
    assert isinstance(g1, GeometryFM3D)
    assert g1.element_coordinates[0, 0] == pytest.approx(0.6666666666666666)

    # you can get elements in arbitrary order
    g2 = g.isel([1, 0])
    assert isinstance(g2, GeometryFM3D)
    assert g2.element_coordinates[1, 0] == pytest.approx(0.6666666666666666)


def test_basic():
    #     x     y    z
    nc = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (0.5, 1.0, 0.0),  # 2
    ]

    el = [(0, 1, 2)]

    g = GeometryFM2D(nc, el)
    assert g.n_elements == 1
    assert g.n_nodes == 3
    assert g.is_2d
    assert g.is_geo
    assert g.is_tri_only
    assert g.projection == "LONG/LAT"
    assert not g.is_layered
    assert 0 in g.find_index(0.5, 0.5)
    with pytest.raises(ValueError):
        g.find_index(50.0, -50.0)

    assert "nodes: 3" in repr(g)


def test_too_many_elements():
    #     x     y    z
    nc = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (0.5, 1.0, 0.0),  # 2
    ]

    el = [(0, 1, 2, 3)]  # There is no node #3

    with pytest.raises(ValueError) as excinfo:
        GeometryFM2D(nc, el)

    assert "element" in str(excinfo.value).lower()


def test_overset_grid():
    #     x     y    z
    nc = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (0.5, 1.0, 0.0),  # 2
    ]

    el = [(0, 1, 2)]

    proj = "UTM-33"
    g = GeometryFM2D(nc, el, projection=proj)
    grid = g.get_overset_grid(dx=0.5)
    assert grid.nx == 2
    assert grid.ny == 2
    assert grid.projection_string == proj


def test_area():
    #     x     y    z
    nc = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (1.0, 1.0, 0.0),  # 2
        (0.0, 1.0, 0.0),  # 3
    ]

    el = [(0, 1, 2, 3)]

    g = GeometryFM2D(node_coordinates=nc, element_table=el, projection="LONG/LAT")
    assert not g.is_tri_only
    area = g.get_element_area()
    assert len(area) == g.n_elements
    assert area > 0.0


def test_find_index_simple_domain():
    #     x     y    z
    nc = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (1.0, 1.0, 0.0),  # 2
        (0.0, 1.0, 0.0),  # 3
        (0.5, 1.5, 0.0),  # 4
    ]

    el = [(0, 1, 2), (0, 2, 3), (3, 2, 4)]

    g = GeometryFM2D(node_coordinates=nc, element_table=el, projection="LONG/LAT")
    idx = g.find_index(0.5, 0.1)
    assert 0 in idx

    # look for multiple points in the same call
    idx = g.find_index(coords=[(0.5, 0.1), (0.1, 0.5)])

    # TODO checking for subsets can only be done if this is a set
    # assert {0, 1} <= idx
    assert 0 in idx
    assert 1 in idx

    # look for the same points multiple times
    idx = g.find_index(coords=[(0.5, 0.1), (0.5, 0.1)])

    # the current behavior is to return the same index twice
    # but this could be changed to return only unique indices
    assert len(idx) == 2
    assert 0 in idx

    with pytest.raises(OutsideModelDomainError) as ex:
        g.find_index(-0.5, -0.1)

    assert ex.value.x == -0.5
    assert ex.value.y == -0.1
    assert 0 in ex.value.indices


def test_isel_simple_domain():
    #     x     y    z
    nc = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (1.0, 1.0, 0.0),  # 2
        (0.0, 1.0, 0.0),  # 3
        (0.5, 1.5, 0.0),  # 4
    ]

    el = [(0, 1, 2), (0, 2, 3), (3, 2, 4)]

    g = GeometryFM2D(node_coordinates=nc, element_table=el, projection="LONG/LAT")
    gp = g.isel(0)
    assert isinstance(gp, GeometryPoint2D)
    assert gp.projection == g.projection


def test_isel_list_of_indices_simple_domain():
    #     x     y    z
    nc = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (1.0, 1.0, 0.0),  # 2
        (0.0, 1.0, 0.0),  # 3
        (0.5, 1.5, 0.0),  # 4
    ]

    el = [(0, 1, 2), (0, 2, 3), (3, 2, 4)]

    g = GeometryFM2D(node_coordinates=nc, element_table=el, projection="LONG/LAT")
    g1 = g.isel([0, 1])
    assert g1.element_coordinates[0, 0] == pytest.approx(0.6666666666666666)

    g2 = g.isel([1, 0])
    assert g2.element_coordinates[1, 0] == pytest.approx(0.6666666666666666)


def test_plot_mesh():
    #     x     y    z
    nc = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (1.0, 1.0, 0.0),  # 2
        (0.0, 1.0, 0.0),  # 3
    ]

    el = [(0, 1, 2), (0, 2, 3)]

    g = GeometryFM2D(node_coordinates=nc, element_table=el, projection="LONG/LAT")
    assert g.n_elements == 2
    g.plot.mesh()


def test_layered(simple_3d_geom: GeometryFM3D):
    g = simple_3d_geom

    assert g.n_elements == 2
    assert g.n_layers == 2
    assert not g.is_2d

    assert len(g.top_elements) == 1

    # find vertical column
    idx = g.find_index(x=0.9, y=0.1)
    assert len(idx) == 2

    # subset
    g2 = g.isel(idx)
    assert g2.n_elements == 2

    assert "elements: 2" in repr(g2)
    assert "layers: 2" in repr(g2)


def test_contains_complex_geometry():
    msh = mikeio.open("tests/testdata/gulf.mesh")

    points = [
        [300_000, 3_200_000],
        [400_000, 3_000_000],
        [800_000, 2_750_000],
        [1_200_000, 2_700_000],
    ]

    res = msh.geometry.contains(points)

    assert all(res)

    res2 = msh.geometry.contains(points, strategy="shapely")
    assert all(res2)


def test_find_index_in_highres_quad_area():
    dfs = mikeio.open("tests/testdata/coastal_quad.dfsu")

    pts = [(439166.047, 6921703.975), (439297.166, 6921728.645)]

    idx = dfs.geometry.find_index(coords=pts)

    assert len(idx) == 2
    for i in idx:
        assert i >= 0


def test_equality():
    nc = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (1.0, 1.0, 0.0),  # 2
        (0.0, 1.0, 0.0),  # 3
    ]

    el = [(0, 1, 2), (0, 2, 3)]

    g = GeometryFM2D(node_coordinates=nc, element_table=el, projection="LONG/LAT")
    g2 = GeometryFM2D(node_coordinates=nc, element_table=el, projection="LONG/LAT")

    assert g == g2

    g3 = GeometryFM2D(node_coordinates=nc, element_table=el, projection="UTM-33")
    assert g != g3


def test_equality_shifted_coords():
    nc1 = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (1.0, 1.0, 0.0),  # 2
        (0.0, 1.0, 0.0),  # 3
    ]

    el = [(0, 1, 2), (0, 2, 3)]
    g = GeometryFM2D(node_coordinates=nc1, element_table=el, projection="LONG/LAT")

    nc2 = [
        (0.1, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (1.0, 1.0, 0.0),  # 2
        (0.1, 1.0, 0.0),  # 3
    ]

    g2 = GeometryFM2D(node_coordinates=nc2, element_table=el, projection="LONG/LAT")
    assert g != g2


def test_da_boundary_polygon() -> None:
    dfs = mikeio.Dfsu2DH("tests/testdata/FakeLake.dfsu")
    bnd = dfs.geometry.boundary_polygon
    assert len(bnd.interiors) == 1
