from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing import overload
from numpy.typing import NDArray

from numpy import float64
import numpy as np

from lirics.leibniz import function_derivative

# TODO : type annotations in methods


def circle_slope(angle):
    """Returns slope of circle for specified angle."""
    return -np.cos(angle)/np.sin(angle)


@dataclass
class Vane(ABC):

    start_radius: float
    end_radius: float
    thickness: float

    total_radius_range: NDArray[float64] = field(init=False)
    angular_width: float = field(init=False)

    def __post_init__(self):
        self.total_radius_range = np.linspace(
            self.start_radius, self.end_radius, PARTITIONS)
        self.angular_width = self.equation(self.end_radius)

    @overload
    def equation(self, radius: float) -> float:
        ...

    @overload
    def equation(self, radius: NDArray[float64]) -> NDArray[float64]:
        ...

    @abstractmethod
    def equation(self, radius):
        """Represents vane's midline equation, returns angular coordinate that
        corresponds to given radius."""
        raise

    def slope(self, radius, step=0.5e-3):
        """Computes vane's midline slope for given radial coordinate with
        central finite difference approximation of derivative."""
        return function_derivative(self.equation, radius, step)

    def length(self, radius_range: NDArray[float64] | None = None) -> NDArray[float64]:
        """Computes length of vane's with numerical trapezoid integration over provided
        array of radius values."""

        if radius_range is None:
            radius_range = self.total_radius_range

        slope = self.slope(radius_range)
        func = np.sqrt(1+radius_range**2*slope**2)
        length = np.trapezoid(func, radius_range, axis=0)

        return length

    def area(self, radius_range):
        """Computes approximate area occupied by vane with trapezoid integration
        over provided array of radius values. Area is computeda as length multiplied by
        thickness."""

        length = self.length(radius_range)

        return length*self.thickness

    @overload
    def adjacent_angle(self, radius: float) -> float:
        ...

    @overload
    def adjacent_angle(self, radius: NDArray[float64]) -> NDArray[float64]:
        ...

    def adjacent_angle(self, radius):
        """Computes angle between vane's tangent and circle's tangent for given radius."""

        vane_angle = self.equation(radius)
        vane_slope_angle = np.arctan(self.slope(radius))
        circle_slope_angle = np.arctan(circle_slope(vane_angle))

        # circle slope will be negative in our case and arctan returns angles in
        # -pi/2 to pi/2 range, in order to get correct value we must subtract angle
        # corresponding to circle clope from pi/2

        return (np.pi/2 - circle_slope_angle) - vane_slope_angle


@dataclass
class StraightVane(Vane):

    @overload
    def equation(self, radius: float) -> float:
        ...

    @overload
    def equation(self, radius: NDArray[float64]) -> NDArray[float64]:
        ...

    def equation(self, radius):
        return np.zeros_like(radius)

    @overload
    def slope(self, radius: float) -> float:
        ...

    @overload
    def slope(self, radius: NDArray[float64]) -> NDArray[float64]:
        ...

    def slope(self, radius):
        return np.zeros_like(radius)

    def length(self, from_radius=None, to_radius=None):

        if from_radius is None:
            from_radius = self.start_radius
        if to_radius is None:
            to_radius = self.end_radius

        return to_radius - from_radius

    def adjacent_angle(self, radius):
        return np.pi/2*np.ones_like(radius)


@dataclass
class ArchVane(Vane):

    transition_radius: float
    end_adjacent_angle: float

    arch_radius: float = field(init=False)
    distance_to_arch_center: float = field(init=False)
    approximation_coeffs: tuple[float, float, float] = field(init=False)

    def __post_init__(self):
        """Computes arch radius and distance to arch center with parameters provided to
        initializer."""
        self.arch_radius = (
            (self.end_radius**2-self.transition_radius**2) /
            (2*self.end_radius*np.cos(self.end_adjacent_angle)))

        self.distance_to_arch_center = np.sqrt(
            self.transition_radius**2+self.arch_radius**2)

    @overload
    def equation(self, radius: float) -> float:
        ...

    @overload
    def equation(self, radius: NDArray[float64]) -> NDArray[float64]:
        ...

    def equation(self, radius):
        angle = np.zeros_like(radius)

        after_transition = (radius >= self.transition_radius)
        radius_after_transition = radius[after_transition]

        angle[after_transition] = (
            np.arcsin(self.arch_radius/self.distance_to_arch_center) -
            np.arccos((radius_after_transition**2 + self.distance_to_arch_center**2 -
                       self.arch_radius**2) /
                      (2*radius_after_transition*self.distance_to_arch_center)))

        return angle

    def compute_approximation_coeffs(self):
        """Computes coefficients for polynomial approximation of vane equation.
        Coefficients are stored in dedicated attribute."""

        arch_middle_radius = 1/2*(self.transition_radius+self.end_radius)

        matrix = np.array([
            [1, self.transition_radius**2, self.transition_radius**4],
            [1, arch_middle_radius**2, arch_middle_radius**4],
            [1, self.end_radius**2, self.end_radius**4]
        ])

        free_vector = np.array(
            [0,
             self.equation(arch_middle_radius),
             self.equation(self.end_radius)]
        )

        self.approximation_coeffs = tuple(
            np.linalg.solve(matrix, free_vector).tolist())

    def approximation(self, radius):
        """Represents an approcimation of vane's equtaion with polynomial expression"""

        a, b, c = self.approximation_coeffs
        angle = a + b*radius**2 + c*radius**4

        return angle


@dataclass
class Impeller:

    length: float
    hub_radius: float
    rim_radius: float

    number_of_cells: float

    vane: Vane

    angular_width_of_cell: float = field(init=False)
    total_area_of_cell: float = field(init=False)
    total_volume_of_cell: float = field(init=False)

    def __post_init__(self):
        """Computes angular width of cell from number of cells, total cell area and total
        cell volume."""

        self.angular_width_of_cell = 2*np.pi/self.number_of_cells

        self.total_radius_range = np.linspace(
            self.hub_radius, self.rim_radius, PARTITIONS)

        self.total_area = self.area(self.total_radius_range)
        self.total_volume = self.total_area*self.length

    @overload
    def clutter_coeff(self, radius: float) -> float:
        ...

    @overload
    def clutter_coeff(self, radius: NDArray[float64]) -> NDArray[float64]:
        ...

    def clutter_coeff(self, radius):
        """Computes cluttering coefficient for given radius to account for radial
        flow area reduction due to the prescence of vanes."""

        adjacent_angle = self.vane.adjacent_angle(radius)
        angular_width = self.angular_width_of_cell
        thickness = self.vane.thickness

        coeff = 1 - thickness/(angular_width*radius*np.sin(adjacent_angle))

        return coeff

    def area_of_cell(self, from_radius=None, to_radius=None, partitions=100):
        """Computes area of cell bounded by given radius values and
        number of partitions."""

        if from_radius is None:
            from_radius = self.hub_radius
        if to_radius is None:
            to_radius = self.rim_radius

        radius = np.linspace(from_radius, to_radius, partitions)
        func = 1/2*radius**2*self.angular_width_of_cell
        pure_cell_area = np.trapezoid(func, radius, axis=0)

        vane_area = self.vane.area(from_radius, to_radius, partitions)

        area = pure_cell_area - vane_area

        return area

    def volume_of_cell(self, from_radius=None, to_radius=None, partitions=100):
        """Computes volume of cell bounded by given radius values and
        number of partitions."""

        area = self.area_of_cell(from_radius, to_radius, partitions)

        return area*self.length

    @overload
    def flow_area(self, radius: float) -> float64:
        ...

    @overload
    def flow_area(self, radius: NDArray[float64]) -> NDArray[float64]:
        ...

    def flow_area(self, radius):
        """Computes area of radial flow in the cell for given radius"""

        clutter_coeff = self.clutter_coeff(radius)
        area = self.length*self.angular_width_of_cell*radius*clutter_coeff

        return area


@dataclass
class Case(ABC):

    length: float

    @overload
    def equation(self, angle: float) -> float:
        ...

    @overload
    def equation(self, angle: NDArray[float64]) -> NDArray[float64]:
        ...

    @abstractmethod
    def equation(self, angle):
        """Represents case's equation, returns radius-vector length for given angle"""
        raise

    def area_of_sector(self, from_angle, to_angle, base_radius=0.0, partitions=100):
        """Computes area of sector in case bounded by base radius and two angular
        coordinates"""

        angles = np.linspace(from_angle, to_angle, partitions)
        case_radius_vector = self.equation(angles)

        base_area = np.trapezoid(1/2*angles*base_radius**2, angles)
        shape_area = np.trapezoid(1/2*angles*case_radius_vector**2, angles)

        return shape_area - base_area

    def volume_of_sector(self, from_angle, to_angle, base_radius=0.0, partitions=100):
        """Computes volume of sector in case bounded by base radius and two angular
        coordinates"""

        area = self.area_of_sector(
            from_angle, to_angle, base_radius, partitions)

        return area*self.length


@dataclass
class CircleCase(Case):

    excentricity: float
    radius: float

    @overload
    def equation(self, angle: float) -> float:
        ...

    @overload
    def equation(self, angle: NDArray[float64]) -> NDArray[float64]:
        ...

    def equation(self, angle):

        radius_vector = self.excentricity*np.cos(angle) + np.sqrt(
            self.excentricity**2*np.cos(angle) + self.radius**2)

        return radius_vector


@dataclass
class EllipticCase(Case):

    major_semiaxis: float
    minor_semiaxis: float

    @overload
    def equation(self, angle: float) -> float:
        ...

    @overload
    def equation(self, angle: NDArray[float64]) -> NDArray[float64]:
        ...

    def equation(self, angle):

        radius_vector = (
            (np.cos(angle)/self.minor_semiaxis)**2 +
            (np.sin(angle)/self.major_semiaxis)**2)**-0.5

        return radius_vector


def compute_total_sector_area(
        sector_geometry: tuple[Vane, Cell, Frame],
        angle: float, partitions=PARTITIONS) -> NDArray[float64]:

    vane, impeller, frame = sector_geometry

    width_of_vane = vane.angular_width
    half_width_of_cell = impeller.angular_width/2

    tip_angle = angle + width_of_vane

    start_angle = tip_angle - half_width_of_cell
    end_angle = tip_angle + half_width_of_cell

    angle_range = np.linspace(start_angle, end_angle, partitions)

    frame_sector_area = frame.sector_area(
        angle_range, impeller.rim_radius)
    cell_area = impeller.total_area

    total_sector_area = frame_sector_area + cell_area

    return total_sector_area


def compute_total_sector_volume(
        sector_geometry: tuple[Vane, Cell, Frame],
        angle: float, partitions=PARTITIONS) -> NDArray[float64]:

    total_sector_area = compute_total_sector_area(
        sector_geometry, angle, partitions)

    # Take impeller's length for computations
    length = sector_geometry[1].length

    return length*total_sector_area


def vane_path(
    vane: Vane, from_radius: float, to_radius: float,
    angular_shift: float = 0.0, partitions: int = PARTITIONS
) -> tuple[NDArray[float64], NDArray[float64]]:

    radius = np.linspace(from_radius, to_radius, partitions)
    angle = vane.equation(radius)+angular_shift

    return radius, angle


def arch_path(
    radius: float, from_angle: float, to_angle: float,
    partition: int = PARTITIONS
) -> tuple[NDArray[float64], NDArray[float64]]:

    angle = np.linspace(from_angle, to_angle, partition)

    return np.ones_like(angle)*radius, angle
