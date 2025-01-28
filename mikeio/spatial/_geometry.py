from abc import ABC, abstractmethod

from dataclasses import dataclass
from typing import Any, Sequence

from mikecore.Projections import MapProjection


@dataclass
class BoundingBox:
    """Bounding box for spatial data."""

    left: float
    bottom: float
    right: float
    top: float

    def __post_init__(self) -> None:
        if self.left > self.right:
            raise ValueError(
                f"Invalid x axis, left: {self.left} must be smaller than right: {self.right}"
            )

        if self.bottom > self.top:
            raise ValueError(
                f"Invalid y axis, bottom: {self.bottom} must be smaller than top: {self.top}"
            )

    def overlaps(self, other: "BoundingBox") -> bool:
        """Check if two bounding boxes overlap."""
        return not (
            self.left > other.right
            or self.bottom > other.top
            or self.right < other.left
            or self.top < other.bottom
        )

    @staticmethod
    def parse(
        values: "BoundingBox" | Sequence[float],
    ) -> "BoundingBox":
        match values:
            case BoundingBox():
                bbox = values
            case left, bottom, right, top:
                bbox = BoundingBox(left, bottom, right, top)
            case _:
                raise ValueError(
                    "values must be a bounding box of coordinates e.g. (-10.0, 10.0 20.0, 30.0)"
                )
        return bbox


class _Geometry(ABC):
    def __init__(self, projection: str = "LONG/LAT") -> None:
        if not MapProjection.IsValid(projection):
            raise ValueError(f"{projection=} is not a valid projection string")

        self._projstr = projection

    @property
    def projection_string(self) -> str:
        """The projection string."""
        return self._projstr

    @property
    def projection(self) -> str:
        """The projection."""
        return self._projstr

    @property
    def is_geo(self) -> bool:
        """Are coordinates geographical (LONG/LAT)?"""
        return self._projstr == "LONG/LAT"

    @property
    def is_local_coordinates(self) -> bool:
        """Are coordinates relative (NON-UTM)?"""
        return self._projstr == "NON-UTM"

    @property
    @abstractmethod
    def ndim(self) -> int:
        pass

    @property
    @abstractmethod
    def default_dims(self) -> tuple[str, ...]:
        pass


class GeometryUndefined(_Geometry):
    def __repr__(self) -> str:
        return "GeometryUndefined()"

    @property
    def ndim(self) -> int:
        raise NotImplementedError()

    @property
    def default_dims(self) -> tuple[str, ...]:
        raise NotImplementedError()


class GeometryPoint2D(_Geometry):
    def __init__(self, x: float, y: float, projection: str = "LONG/LAT"):
        super().__init__(projection)
        self.x = x
        self.y = y

    @property
    def default_dims(self) -> tuple[str, ...]:
        return ()

    def __repr__(self) -> str:
        return f"GeometryPoint2D(x={self.x}, y={self.y})"

    # TODO should we use wkt here
    # def __str__(self) -> str:
    #    return self.wkt

    @property
    def wkt(self) -> str:
        return f"POINT ({self.x} {self.y})"

    @property
    def ndim(self) -> int:
        """Geometry dimension."""
        return 0

    def __len__(self) -> int:
        return 1

    def to_shapely(self) -> Any:
        from shapely.geometry import Point

        return Point(self.x, self.y)


class GeometryPoint3D(_Geometry):
    def __init__(self, x: float, y: float, z: float, projection: str = "LONG/LAT"):
        super().__init__(projection)

        self.x = x
        self.y = y
        self.z = z

    def __repr__(self) -> str:
        return f"GeometryPoint3D(x={self.x}, y={self.y}, z={self.z})"

    @property
    def default_dims(self) -> tuple[str, ...]:
        return ()

    @property
    def wkt(self) -> str:
        return f"POINT Z ({self.x} {self.y} {self.z})"

    @property
    def ndim(self) -> int:
        return 0

    def __len__(self) -> int:
        return 1

    def to_shapely(self) -> Any:
        from shapely.geometry import Point

        return Point(self.x, self.y, self.z)
