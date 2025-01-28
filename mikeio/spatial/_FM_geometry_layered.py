from __future__ import annotations
from functools import cached_property
from pathlib import Path

from typing import Any, Iterable, Literal, Sequence

from matplotlib.axes import Axes
import numpy as np
from mikecore.DfsuFile import DfsuFileType
from shapely import Polygon


from ._FM_geometry import GeometryFM2D, _GeometryFM, _GeometryFMPlotter
from ._geometry import GeometryPoint3D

from ._FM_utils import _plot_vertical_profile

from ._utils import _relative_cumulative_distance

Layer = Literal["all", "bottom", "top"]


class _GeometryFMLayered(_GeometryFM):
    def __init__(
        self,
        *,
        node_coordinates: np.ndarray,
        element_table: np.ndarray | list[Sequence[int]] | list[np.ndarray],
        codes: np.ndarray | None = None,
        projection: str = "LONG/LAT",
        dfsu_type: DfsuFileType = DfsuFileType.Dfsu3DSigma,
        element_ids: np.ndarray | None = None,
        node_ids: np.ndarray | None = None,
        n_layers: int = 1,  # at least 1 layer
        n_sigma: int | None = None,
        validate: bool = True,
        reindex: bool = False,
    ) -> None:
        super().__init__(
            node_coordinates=node_coordinates,
            element_table=element_table,
            codes=codes,
            projection=projection,
            dfsu_type=dfsu_type,
            element_ids=element_ids,
            node_ids=node_ids,
            validate=validate,
            reindex=reindex,
        )

        self._n_layers = n_layers
        self._n_sigma = n_sigma if n_sigma is not None else n_layers

        # Lazy properties
        self._bot_elems: np.ndarray | None = None
        self._e2_e3_table: np.ndarray | None = None
        self._2d_ids: np.ndarray | None = None
        self._layer_ids: np.ndarray | None = None

    def __repr__(self) -> str:
        return (
            f"Flexible Mesh Geometry: {self._type.name}\n"
            f"number of nodes: {self.n_nodes}\n"
            f"number of elements: {self.n_elements}\n"
            f"number of layers: {self.n_layers}\n"
            f"number of sigma layers: {self.n_sigma_layers}\n"
            f"projection: {self.projection_string}"
        )

    @cached_property
    def geometry2d(self) -> GeometryFM2D:
        return self.to_2d_geometry()

    def isel(
        self, idx: Sequence[int] | np.ndarray, keepdims: bool = False, **kwargs: Any
    ) -> GeometryFM3D | GeometryPoint3D | GeometryFM2D | GeometryFMVerticalColumn:
        return self.elements_to_geometry(elements=idx, keepdims=keepdims)

    def elements_to_geometry(
        self,
        elements: int | Sequence[int] | np.ndarray,
        node_layers: Layer = "all",
        keepdims: bool = False,
    ) -> GeometryFM3D | GeometryPoint3D | GeometryFM2D | GeometryFMVerticalColumn:
        sel_elements: list[int]

        if isinstance(elements, (int, np.integer)):
            sel_elements = [elements]
        else:
            sel_elements = list(elements)
        if len(sel_elements) == 1 and not keepdims:
            x, y, z = self.element_coordinates[sel_elements.pop(), :]

            return GeometryPoint3D(x=x, y=y, z=z, projection=self.projection)

        # create new geometry
        new_type = self._type

        layers_used = self.layer_ids[sel_elements]
        unique_layer_ids = np.unique(layers_used)
        n_layers = len(unique_layer_ids)

        if n_layers > 1:
            bottom: Layer = "bottom"
            elem_bot = self.get_layer_elements(layers=bottom)
            if np.all(np.isin(sel_elements, elem_bot)):
                n_layers = 1

        if (
            self._type == DfsuFileType.Dfsu3DSigma
            or self._type == DfsuFileType.Dfsu3DSigmaZ
        ) and n_layers == 1:
            new_type = DfsuFileType.Dfsu2D

        if n_layers == 1 and node_layers == "all":
            node_layers = "bottom"

        # extract information for selected elements
        if n_layers == 1:
            elem2d = self.elem2d_ids[sel_elements]
            geom2d = self.geometry2d
            node_ids, elem_tbl = geom2d._get_nodes_and_table_for_elements(elem2d)
            assert len(elem_tbl[0]) <= 4, "Not a 2D element"
            node_coords = geom2d.node_coordinates[node_ids]
            codes = geom2d.codes[node_ids]
            elem_ids = self._element_ids[elem2d]
        else:
            node_ids, elem_tbl = self._get_nodes_and_table_for_elements(
                sel_elements, node_layers=node_layers
            )
            node_coords = self.node_coordinates[node_ids]
            codes = self.codes[node_ids]
            elem_ids = self._element_ids[sel_elements]

        if new_type == DfsuFileType.Dfsu2D:
            return GeometryFM2D(
                node_coordinates=node_coords,
                codes=codes,
                node_ids=node_ids,
                projection=self.projection_string,
                element_table=elem_tbl,
                element_ids=elem_ids,
                dfsu_type=DfsuFileType.Dfsu2D,
                reindex=True,
            )
        else:
            lowest_sigma = self.n_layers - self.n_sigma_layers
            n_sigma = sum(unique_layer_ids >= lowest_sigma)
            if n_layers == len(elem_tbl):
                return GeometryFMVerticalColumn(
                    node_coordinates=node_coords,
                    codes=codes,
                    node_ids=node_ids,
                    projection=self.projection_string,
                    element_table=elem_tbl,
                    element_ids=elem_ids,
                    dfsu_type=self._type,
                    reindex=True,
                    n_layers=n_layers,
                    n_sigma=n_sigma,
                )
            else:
                klass = self.__class__
                return klass(  # type: ignore
                    node_coordinates=node_coords,
                    codes=codes,
                    node_ids=node_ids,
                    projection=self.projection_string,
                    element_table=elem_tbl,
                    element_ids=elem_ids,
                    dfsu_type=self._type,
                    reindex=True,
                    n_layers=n_layers,
                    n_sigma=n_sigma,
                )

    @cached_property
    def element_coordinates(self) -> np.ndarray:
        """Center coordinates of each element."""
        return self._calc_element_coordinates(maxnodes=8)

    def _get_nodes_and_table_for_elements(
        self,
        elements: Sequence[int] | np.ndarray,
        node_layers: Layer = "all",
    ) -> tuple[Any, Any]:
        """list of nodes and element table for a list of elements.

        Parameters
        ----------
        elements : np.array(int)
            array of element ids
        node_layers : str, optional
            for 3D files 'all', 'bottom' or 'top' nodes
            of each element, by default 'all'

        Returns
        -------
        np.array(int)
            array of node ids (unique)
        list(list(int))
            element table with a list of nodes for each element

        """
        elem_tbl = np.empty(len(elements), dtype=np.dtype("O"))
        if (node_layers == "all") or self.is_2d:
            for j, eid in enumerate(elements):
                elem_tbl[j] = np.asarray(self.element_table[eid])

        else:
            # 3D => 2D
            for j, eid in enumerate(elements):
                elem_nodes = np.asarray(self.element_table[eid])
                nn = len(elem_nodes)
                halfn = int(nn / 2)
                if node_layers == "bottom":
                    elem_nodes = elem_nodes[:halfn]
                if node_layers == "top":
                    elem_nodes = elem_nodes[halfn:]
                elem_tbl[j] = elem_nodes

        nodes = np.unique(np.hstack(elem_tbl))  # type: ignore
        return nodes, elem_tbl

    def to_2d_geometry(self) -> GeometryFM2D:
        """extract 2d geometry from 3d geometry.

        Returns
        -------
        GeometryFM2D
            2d geometry

        """
        # extract information for selected elements
        match self._type:
            case DfsuFileType.Dfsu3DSigmaZ:
                elem_ids = self.top_elements
            case DfsuFileType.Dfsu3DSigma:
                elem_ids = self.bottom_elements
            case _:
                raise NotImplementedError(
                    f"Conversion to 2D not implemented for {self._type}"
                )

        node_ids, elem_tbl = self._get_nodes_and_table_for_elements(
            elem_ids, node_layers="bottom"
        )
        node_coords = self.node_coordinates[node_ids]
        codes = self._codes[node_ids]

        geom = GeometryFM2D(
            node_coordinates=node_coords,
            codes=codes,
            node_ids=node_ids,
            projection=self.projection_string,
            element_table=elem_tbl,
            element_ids=elem_ids,
            validate=False,
            dfsu_type=DfsuFileType.Dfsu2D,
            reindex=True,
        )

        # TODO do this before creating the geometry

        # TODO extract method
        # Fix z-coordinate for sigma-z:
        if self._type == DfsuFileType.Dfsu3DSigmaZ:
            zn = geom.node_coordinates[:, 2].copy()
            for j, elem_nodes in enumerate(geom.element_table):
                elem_nodes3d = self.element_table[self.bottom_elements[j]]
                for jn in range(len(elem_nodes)):
                    znj_3d = self.node_coordinates[elem_nodes3d[jn], 2]
                    zn[elem_nodes[jn]] = min(zn[elem_nodes[jn]], znj_3d)
            geom.node_coordinates[:, 2] = zn

        return geom

    @cached_property
    def n_elements(self) -> int:
        """Number of 3d elements."""
        return len(self.element_table)

    @property
    def n_nodes(self) -> int:
        return len(self.node_coordinates)

    @property
    def is_2d(self) -> bool:
        return False

    @property
    def ndim(self) -> int:
        return 3

    @property
    def is_layered(self) -> bool:
        return True

    @cached_property
    def layer_ids(self) -> np.ndarray:
        """The layer number (0=bottom, 1, 2, ...) for each 3d element."""
        if self._layer_ids is None:
            res = self._get_2d_to_3d_association()
            self._e2_e3_table = res[0]
            self._2d_ids = res[1]
            self._layer_ids = res[2]
        return self._layer_ids

    @property
    def n_layers(self) -> int:
        """Maximum number of layers."""
        return self._n_layers

    @property
    def n_sigma_layers(self) -> int:
        """Number of sigma layers."""
        return self._n_sigma

    @property
    def n_z_layers(self) -> int:
        """Maximum number of z-layers."""
        return self.n_layers - self.n_sigma_layers

    @cached_property
    def top_elements(self) -> np.ndarray:
        """List of 3d element ids of surface layer."""
        # note: if subset of elements is selected then this cannot be done!

        # fast path if no z-layers
        if self.n_z_layers == 0:
            return np.arange(
                start=self.n_sigma_layers - 1,
                stop=self.n_elements,
                step=self.n_sigma_layers,
            )
        else:
            # slow path
            return self._find_top_layer_elements(self.element_table)

    def _elements_in_area(
        self, area: Sequence[tuple[float, float]] | Sequence[float]
    ) -> np.ndarray:
        """Find element ids of elements inside area."""
        idx2d = self.geometry2d._elements_in_area(area)
        if len(idx2d) > 0:
            return np.hstack(self.e2_e3_table[idx2d])
        else:
            return np.array([], dtype=int)

    @staticmethod
    def _find_top_layer_elements(elementTable: np.ndarray) -> np.ndarray:
        """Find element indices (zero based) of the elements being the upper-most element
        in its column.
        Each column is identified by matching node id numbers. For 3D elements the
        last half of the node numbers of the bottom element must match the first half
        of the node numbers in the top element. For 2D vertical elements the order of
        the node numbers in the bottom element (last half number of nodes) are reversed
        compared to those in the top element (first half number of nodes).
        To find the number of elements in each column, assuming the result
        is stored in res:
        For the first column it is res[0]+1.
        For the i'th column, it is res[i]-res[i-1].
        :returns: A list of element indices of top layer elements.
        """
        topLayerElments = []

        # Find top layer elements by matching the number numers of the last half of elmt i
        # with the first half of element i+1.
        # Elements always start from the bottom, and the element of one columne are following
        # each other in the element table.
        for i in range(len(elementTable) - 1):
            elmt1 = elementTable[i]
            elmt2 = elementTable[i + 1]

            if len(elmt1) != len(elmt2):
                # elements with different number of nodes can not be on top of each other,
                # so elmt2 must be another column, and elmt1 must be a top element
                topLayerElments.append(i)
                continue

            # Number of nodes in a 2D element
            elmt2DSize = len(elmt1) // 2

            for j in range(elmt2DSize):
                if elmt2DSize > 2:
                    if elmt1[j + elmt2DSize] != elmt2[j]:
                        # At least one node number did not match
                        # so elmt2 must be another column, and elmt1 must be a top element
                        topLayerElments.append(i)
                        break
                else:
                    # for 2D vertical profiles the nodes in the element on the
                    # top is in reverse order of those in the bottom.
                    if elmt1[j + elmt2DSize] != elmt2[(elmt2DSize - 1) - j]:
                        # At least one node number did not match
                        # so elmt2 must be another column, and elmt1 must be a top element
                        topLayerElments.append(i)
                        break

        # The last element will always be a top layer element
        topLayerElments.append(len(elementTable) - 1)

        return np.array(topLayerElments, dtype=np.int32)

    @cached_property
    def n_layers_per_column(self) -> np.ndarray:
        """List of number of layers for each column."""
        top_elems = self.top_elements
        n = len(top_elems)
        tmp = top_elems.copy()
        tmp[0] = -1
        tmp[1:n] = top_elems[0 : (n - 1)]
        n_layers_column = top_elems - tmp
        return n_layers_column

    @cached_property
    def bottom_elements(self) -> np.ndarray:
        """List of 3d element ids of bottom layer."""
        return self.top_elements - self.n_layers_per_column + 1

    def get_layer_elements(self, layers: int | Layer | Sequence[int]) -> np.ndarray:
        """3d element ids for one (or more) specific layer(s).

        Parameters
        ----------
        layers : int or list(int)
            layer between 0 (bottom) and n_layers-1 (top)
            (can also be negative counting from -1 at the top layer)

        Returns
        -------
        np.array(int)
            element ids

        """
        if isinstance(layers, str):
            if layers == "top":
                return self.top_elements
            elif layers == "bottom":
                return self.bottom_elements
            else:
                raise ValueError(
                    f"layers '{layers}' not recognized ('top', 'bottom' or integer)"
                )

        if not np.isscalar(layers):
            assert isinstance(layers, Iterable)
            elem_ids = []
            for layer in layers:
                assert isinstance(layer, int)
                elem_ids.append(self.get_layer_elements(layer))
            elem_ids = np.concatenate(elem_ids, axis=0)
            return np.sort(elem_ids)

        n_lay = self.n_layers

        assert isinstance(layers, int)
        if layers < (-n_lay) or layers >= n_lay:
            raise Exception(
                f"Layer {layers!r} not allowed; must be between -{n_lay} and {n_lay-1}"
            )

        if layers < 0:
            layers = layers + n_lay

        return self._element_ids[self.layer_ids == layers]

    @property
    def e2_e3_table(self) -> np.ndarray:
        """The 2d-to-3d element connectivity table for a 3d object."""
        # e2_e3, 2d_ids and layer_ids are all set at the same time

        if self._e2_e3_table is None:
            res = self._get_2d_to_3d_association()
            self._e2_e3_table = res[0]
            self._2d_ids = res[1]
            self._layer_ids = res[2]
        return self._e2_e3_table

    @property
    def elem2d_ids(self) -> np.ndarray:
        """The associated 2d element id for each 3d element."""
        if self._2d_ids is None:
            res = self._get_2d_to_3d_association()
            self._e2_e3_table = res[0]
            self._2d_ids = res[1]
            self._layer_ids = res[2]
        return self._2d_ids

    def _get_2d_to_3d_association(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        e2_to_e3 = []  # for each 2d element: the corresponding 3d element ids from bot to top
        index2d = []  # for each 3d element: the associated 2d element id
        layerid = []  # for each 3d element: the associated layer number
        n2d = len(self.top_elements)
        topid = self.top_elements
        botid = self.bottom_elements
        # layer_ids = 0, 1, 2...
        global_layer_ids = np.arange(self.n_layers)
        for j in range(n2d):
            col = np.arange(botid[j], topid[j] + 1)

            e2_to_e3.append(col)
            for _ in col:
                index2d.append(j)

            n_local_layers = len(col)
            local_layers = global_layer_ids[-n_local_layers:]
            for ll in local_layers:
                layerid.append(ll)

        e2_to_e3 = np.array(e2_to_e3, dtype=object)
        index2d = np.array(index2d)
        layerid = np.array(layerid)
        return e2_to_e3, index2d, layerid

    def _z_idx_in_column(self, e3_col: np.ndarray, z: np.ndarray) -> np.ndarray:
        dz = self._dz[e3_col]
        z_col = self.element_coordinates[e3_col, 2]
        z_face = np.append(z_col - dz / 2, z_col[-1] + dz[-1] / 2)
        if z < z_face[0] or z > z_face[-1]:
            xy = tuple(self.element_coordinates[e3_col[0], :2])
            raise ValueError(
                f"z value '{z}' is outside water column [{z_face[0]},{z_face[-1]}] in point x,y={xy}"
            )
        idx = np.searchsorted(z_face, z) - 1
        return idx

    def _find_elem3d_from_elem2d(
        self, elem2d: int | np.ndarray, z: np.ndarray | float
    ) -> np.ndarray:
        """Find 3d element ids from 2d element ids and z-values."""
        # TODO: coordinate with _find_3d_from_2d_points()

        elem2d = [elem2d] if np.isscalar(elem2d) else elem2d
        elem2d = np.asarray(elem2d)
        z_vec = np.full(elem2d.shape, fill_value=z) if np.isscalar(z) else z

        elem3d = np.full_like(elem2d, fill_value=-1)
        for j, e2 in enumerate(elem2d):
            idx_3d = np.hstack(self.e2_e3_table[e2])
            elem3d[j] = idx_3d[self._z_idx_in_column(idx_3d, z_vec[j])]  # type: ignore

            # z_col = self.element_coordinates[idx_3d, 2]
            # elem3d[j] = (np.abs(z_col - z_vec[j])).argmin()  # nearest
        return elem3d

    # def _find_3d_from_2d_points(self, elem2d, z=None, layer=None):

    #     was_scalar = np.isscalar(elem2d)
    #     if was_scalar:
    #         elem2d = np.array([elem2d])
    #     else:
    #         orig_shape = elem2d.shape
    #         elem2d = np.reshape(elem2d, (elem2d.size,))

    #     if (layer is None) and (z is None):
    #         # return top element
    #         idx = self.top_elements[elem2d]  # TODO: return whole column instead

    #     elif layer is None:
    #         idx = np.zeros_like(elem2d)
    #         if np.isscalar(z):
    #             z = z * np.ones_like(elem2d, dtype=float)
    #         elem3d = self.e2_e3_table[elem2d]
    #         for j, row in enumerate(elem3d):
    #             zc = self.element_coordinates[row, 2]
    #             d3d = np.abs(z[j] - zc)
    #             idx[j] = row[d3d.argsort()[0]]

    #     elif z is None:
    #         if 0 <= layer <= self.n_z_layers - 1:
    #             idx = np.zeros_like(elem2d)
    #             elem3d = self.e2_e3_table[elem2d]
    #             for j, row in enumerate(elem3d):
    #                 try:
    #                     layer_ids = self.layer_ids[row]
    #                     id = row[list(layer_ids).index(layer)]
    #                     idx[j] = id
    #                 except IndexError:
    #                     raise IndexError(
    #                         f"Layer {layer} not present for 2d element {elem2d[j]}"
    #                     )
    #         else:
    #             # sigma layer
    #             idx = self.get_layer_elements(layer)[elem2d]

    #     else:
    #         raise ValueError("layer and z cannot both be supplied!")

    #     if was_scalar:
    #         idx = idx[0]
    #     else:
    #         idx = np.reshape(idx, orig_shape)

    #     return idx

    @cached_property
    def _dz(self) -> np.ndarray:
        """Height of each 3d element (using static zn information)."""
        return self._calc_dz()

    def _calc_dz(self) -> np.ndarray:
        """Height of 3d elements using static or dynamic zn information."""
        element_table = self.element_table
        n_elements = len(element_table)

        zn = self.node_coordinates[:, 2]
        zn_is_2d = len(zn.shape) == 2
        shape = (zn.shape[0], n_elements) if zn_is_2d else (n_elements,)
        dz = np.full(shape=shape, fill_value=np.nan)

        # static zn
        for j in range(n_elements):
            nodes = element_table[j]
            halfn = int(len(nodes) / 2)
            z_bot = np.mean(zn[nodes[:halfn]])
            z_top = np.mean(zn[nodes[halfn:]])
            dz[j] = z_top - z_bot

        return dz


class GeometryFM3D(_GeometryFMLayered):
    """Flexible 3d mesh geometry."""

    def __init__(
        self,
        *,
        node_coordinates: np.ndarray,
        element_table: np.ndarray | list[Sequence[int]] | list[np.ndarray],
        codes: np.ndarray | None = None,
        projection: str = "LONG/LAT",
        dfsu_type: DfsuFileType = DfsuFileType.Dfsu3DSigma,
        element_ids: np.ndarray | None = None,
        node_ids: np.ndarray | None = None,
        n_layers: int = 1,  # at least 1 layer
        n_sigma: int | None = None,
        validate: bool = True,
        reindex: bool = False,
    ) -> None:
        super().__init__(
            node_coordinates=node_coordinates,
            element_table=element_table,
            codes=codes,
            projection=projection,
            dfsu_type=dfsu_type,
            element_ids=element_ids,
            node_ids=node_ids,
            n_layers=n_layers,
            n_sigma=n_sigma,
            validate=validate,
            reindex=reindex,
        )
        self.plot = _GeometryFMPlotter(self)

    @property
    def boundary_polygon(self) -> Polygon:
        return self.geometry2d.boundary_polygon

    def contains(self, points: np.ndarray) -> np.ndarray:
        return self.geometry2d.contains(points)

    def to_mesh(self, outfilename: str | Path) -> None:
        return self.geometry2d.to_mesh(outfilename)

    def find_index(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        coords: np.ndarray | None = None,
        area: tuple[float, float, float, float] | None = None,
        layers: int | Layer | Sequence[int] | None = None,
    ) -> np.ndarray:
        if layers is not None:
            idx = self.get_layer_elements(layers)
        else:
            idx = self._element_ids

        if (
            (coords is not None)
            or (x is not None)
            or (y is not None)
            or (z is not None)
        ):
            if area is not None:
                raise ValueError(
                    "Coordinates and area cannot be provided at the same time!"
                )
            if coords is not None:
                coords = np.atleast_2d(coords)
                xy = coords[:, :2]  # type: ignore
                z = coords[:, 2] if coords.shape[1] == 3 else None  # type: ignore
            else:
                xy = np.vstack((x, y)).T  # type: ignore
            idx_2d = self.geometry2d._find_element_2d(coords=xy)
            assert len(idx_2d) == len(xy)
            if z is None:
                idx_3d = np.hstack(self.e2_e3_table[idx_2d])
            else:
                idx_3d = self._find_elem3d_from_elem2d(idx_2d, z)
            idx = np.intersect1d(idx, idx_3d).astype(int)
        elif area is not None:
            idx_area = self._elements_in_area(area)
            idx = np.intersect1d(idx, idx_area)
        elif layers is None:
            raise ValueError(
                "At least one selection argument (x,y,z,coords,area,layers) needs to be provided!"
            )
        return idx


class GeometryFMVerticalProfile(_GeometryFMLayered):
    """Flexible mesh 2d vertical profile geometry."""

    def __init__(
        self,
        node_coordinates: np.ndarray,
        element_table: np.ndarray | list[Sequence[int]] | list[np.ndarray],
        codes: np.ndarray | None = None,
        projection: str = "LONG/LAT",
        dfsu_type: DfsuFileType = DfsuFileType.Dfsu3DSigma,
        element_ids: np.ndarray | None = None,
        node_ids: np.ndarray | None = None,
        n_layers: int = 1,  # at least 1 layer
        n_sigma: int | None = None,
        validate: bool = True,
        reindex: bool = False,
    ) -> None:
        super().__init__(
            node_coordinates=node_coordinates,
            element_table=element_table,
            codes=codes,
            projection=projection,
            dfsu_type=dfsu_type,
            element_ids=element_ids,
            node_ids=node_ids,
            n_layers=n_layers,
            n_sigma=n_sigma,
            validate=validate,
            reindex=reindex,
        )
        self.plot = _GeometryFMVerticalProfilePlotter(self)

    @cached_property
    def relative_element_distance(self) -> np.ndarray:
        ec = self.element_coordinates
        nc0 = self.node_coordinates[0, :2]
        return _relative_cumulative_distance(ec, nc0, is_geo=self.is_geo)

    def get_nearest_relative_distance(self, coords: tuple[float, float]) -> float:
        """For a point near a transect, find the nearest relative distance
        for showing position on transect plot.

        Parameters
        ----------
        coords : [float, float]
            x,y-coordinate of point

        Returns
        -------
        float
            relative distance in meters from start of transect

        """
        xe = self.element_coordinates[:, 0]
        ye = self.element_coordinates[:, 1]
        dd2 = np.square(xe - coords[0]) + np.square(ye - coords[1])
        idx = np.argmin(dd2)
        return self.relative_element_distance[idx]

    def find_index(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        coords: np.ndarray | None = None,
        layers: int | Sequence[int] | Layer | None = None,
    ) -> np.ndarray:
        if layers is not None:
            idx = self.get_layer_elements(layers)
        else:
            idx = self._element_ids

        # select in space
        if (
            (coords is not None)
            or (x is not None)
            or (y is not None)
            or (z is not None)
        ):
            if coords is not None:
                coords = np.atleast_2d(coords)
                xy = coords[:, :2]  # type: ignore
                z = coords[:, 2] if coords.shape[1] == 3 else None  # type: ignore
            else:
                xy = np.vstack((x, y)).T  # type: ignore

            idx_2d = self._find_nearest_element_2d(coords=xy)

            if z is None:
                idx_3d = np.hstack(self.e2_e3_table[idx_2d])
            else:
                idx_3d = self._find_elem3d_from_elem2d(idx_2d, z)
            idx = np.intersect1d(idx, idx_3d)

        return idx

    def _find_nearest_element_2d(self, coords: np.ndarray) -> np.ndarray:
        ec2d = self.element_coordinates[self.top_elements, :2]
        xe, ye = ec2d[:, 0], ec2d[:, 1]
        coords = np.atleast_2d(coords)
        idx = np.zeros(len(coords), dtype=int)
        for j, pt in enumerate(coords):
            x, y = pt[0:2]
            idx[j] = np.argmin((xe - x) ** 2 + (ye - y) ** 2)
        return idx


class GeometryFMVerticalColumn(GeometryFM3D):
    "A 3d geometry with consisting of a single vertical column."

    # TODO: add plotter

    def calc_ze(self, zn: np.ndarray | None = None) -> np.ndarray:
        if zn is None:
            zn = self.node_coordinates[:, 2]
        return self._calc_z_using_idx(zn, self._idx_e)

    def calc_zf(self, zn: np.ndarray | None = None) -> np.ndarray:
        if zn is None:
            zn = self.node_coordinates[:, 2]
        return self._calc_z_using_idx(zn, self._idx_f)

    def _calc_zee(self, zn: np.ndarray | None = None) -> np.ndarray:
        ze = self.calc_ze(zn)
        zf = self.calc_zf(zn)
        if ze.ndim == 1:
            zee = np.insert(ze, 0, zf[0])
            return np.append(zee, zf[-1])
        else:
            return np.hstack(
                (zf[:, 0].reshape((-1, 1)), ze, zf[:, -1].reshape((-1, 1)))
            )

    def _interp_values(
        self, zn: np.ndarray, data: np.ndarray, z: np.ndarray
    ) -> np.ndarray:
        """Interpolate to other z values, allow linear extrapolation."""
        from scipy.interpolate import interp1d  # type: ignore

        opt = {"kind": "linear", "bounds_error": False, "fill_value": "extrapolate"}

        ze = self.calc_ze(zn)
        dati = np.zeros_like(z)
        if zn.ndim == 1:
            dati = interp1d(ze, data, **opt)(z)
        elif zn.ndim == 2:
            for j in range(zn.shape[0]):
                dati[j, :] = interp1d(ze[j, :], data[j, :], **opt)(z[j, :])
        return dati

    @property
    def _idx_f(self) -> np.ndarray:
        nnodes_half = int(len(self.element_table[0]) / 2)
        n_vfaces = self.n_elements + 1
        idx_f = np.zeros((n_vfaces, nnodes_half), dtype=int)
        idx_e = self._idx_e
        idx_f[: self.n_elements, :] = idx_e[:, :nnodes_half]
        idx_f[self.n_elements, :] = idx_e[-1, nnodes_half:]
        return idx_f

    @property
    def _idx_e(self) -> np.ndarray:
        nnodes_per_elem = len(self.element_table[0])
        idx_e = np.zeros((self.n_elements, nnodes_per_elem), dtype=int)

        for j in range(self.n_elements):
            nodes = self.element_table[j]
            for i in range(nnodes_per_elem):
                idx_e[j, i] = nodes[i]

        return idx_e

    def _calc_z_using_idx(self, zn: np.ndarray, idx: np.ndarray) -> np.ndarray:
        if zn.ndim == 1:
            zf = zn[idx].mean(axis=1)
        elif zn.ndim == 2:
            n_steps = zn.shape[0]
            zf = np.zeros((n_steps, idx.shape[0]))
            for step in range(n_steps):
                zf[step, :] = zn[step, idx].mean(axis=1)

        return zf


class _GeometryFMVerticalProfilePlotter:
    def __init__(self, geometry: "GeometryFMVerticalProfile") -> None:
        self.g = geometry

    def __call__(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> Axes:
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        x = self.g.node_coordinates[:, 0]
        y = self.g.node_coordinates[:, 1]
        ax.plot(x, y, **kwargs)
        return ax

    def mesh(self, title: str = "Mesh", edge_color: str = "0.5", **kwargs: Any) -> Axes:
        v = np.full_like(self.g.element_coordinates[:, 0], np.nan)
        return _plot_vertical_profile(
            node_coordinates=self.g.node_coordinates,
            element_table=self.g.element_table,
            values=v,
            is_geo=self.g.is_geo,
            title=title,
            add_colorbar=False,
            edge_color=edge_color,
            cmin=0.0,
            cmax=1.0,
            **kwargs,
        )
