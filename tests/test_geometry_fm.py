import pytest
from mikeio.spatial import GeometryFM2D, GeometryFM3D
from mikeio.exceptions import OutsideModelDomainError
from mikeio.spatial import GeometryPoint2D


@pytest.fixture
def simple_3d_geom():

    nc_2d = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    # bottom first approach
    z = [-2.0, -1.0, 0.0]

    nc = []

    for col in nc_2d:
        for depth in z:
            nc.append((col[0], col[1], depth))

    # TODO constructing a 3d mesh following the convention needs a helper function
    # Arbitrary 3d element tables doesn't work
    el = [(0, 3, 6, 1, 4, 7), (1, 4, 7, 2, 5, 8)]

    g = GeometryFM3D(
        node_coordinates=nc,
        element_table=el,
        projection="LONG/LAT",
        n_layers=2,
        n_sigma=2,
    )

    return g


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


def test_layered(simple_3d_geom: GeometryFM3D) -> None:

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


def test_select_single_layer_preserved_vertical_coordinates(
    simple_3d_geom: GeometryFM3D,
) -> None:
    g = simple_3d_geom

    bot_el = g.bottom_elements
    gb = g.elements_to_geometry(bot_el, keepdims=True)
    assert isinstance(gb, GeometryFM2D)
    assert gb.node_coordinates[0][2] == -2.0

    top_el = g.top_elements

    gt = g.elements_to_geometry(top_el, keepdims=True)
    assert isinstance(gt, GeometryFM2D)
    assert gt.n_elements == 1
    assert gt.node_coordinates[0][2] == -1.0
