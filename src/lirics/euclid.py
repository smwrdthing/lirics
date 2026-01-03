from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing import overload
from numpy.typing import NDArray

from numpy import float64
import numpy as np

from lirics.leibniz import function_derivative

STEP = 0.5e-3
PARTITIONS = 100


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

    def derivative(self, radius: NDArray[float64], step: float = STEP):
        """Computes vane's midline derivative for given radial coordinate with
        central finite difference approximation of derivative."""
        return function_derivative(self.equation, radius, step)

    def slope(self, radius, step=STEP):

        angle = self.equation(radius)
        deriv = self.derivative(radius, step)

        incline = ((np.sin(angle)+radius*deriv*np.cos(angle)) /
                   (np.cos(angle)-radius*deriv*np.sin(angle)))

        return incline

    def length(self, radius_range: NDArray[float64] | None = None) -> NDArray[float64]:
        """Computes length of vane's with numerical trapezoid integration over provided
        array of radius values."""

        if radius_range is None:
            radius_range = self.total_radius_range

        deriv = self.derivative(radius_range)
        func = np.sqrt(1+radius_range**2*deriv**2)
        length = np.trapezoid(func, radius_range, axis=0)

        return length

    def area(self, radius_array: NDArray[float64] | None = None) -> NDArray[float64]:
        """Computes approximate area occupied by vane with trapezoid integration
        over provided array of radius values. Area is computeda as length multiplied by
        thickness."""

        if radius_array is None:
            radius_array = self.total_radius_range

        length = self.length(radius_array)

        return length*self.thickness

    @overload
    def adjacent_angle(self, radius: float, step: float = STEP) -> float:
        ...

    @overload
    def adjacent_angle(
            self, radius: NDArray[float64], step: float = STEP) -> NDArray[float64]:
        ...

    def adjacent_angle(self, radius, step=STEP):
        """Computes angle between vane's tangent and circle's tangent for given radius."""

        vane_angle = self.equation(radius)
        circle_slope_angle = vane_angle + np.pi/2
        vane_slope_angle = np.arctan(self.slope(radius, step))

        return circle_slope_angle - vane_slope_angle


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
    def derivative(self, radius: float) -> float:
        ...

    @overload
    def derivative(self, radius: NDArray[float64]) -> NDArray[float64]:
        ...

    def derivative(self, radius):
        return np.zeros_like(radius)

    def length(self, radius_array: NDArray[float64]) -> NDArray[float64]:
        if radius_array is None:
            radius_array = self.total_radius_range
        return radius_array[-1]-radius_array[0]

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

        super().__post_init__()

    @overload
    def equation(self, radius: float) -> float:
        ...

    @overload
    def equation(self, radius: NDArray[float64]) -> NDArray[float64]:
        ...

    def equation(self, radius):
        angle = np.zeros_like(radius)

        after_transition = (radius >= self.transition_radius)

        angle += after_transition*(
            np.arcsin(self.arch_radius/self.distance_to_arch_center) -
            np.arccos((radius**2 + self.distance_to_arch_center**2 -
                       self.arch_radius**2) /
                      (2*radius*self.distance_to_arch_center)))

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
class Cell:

    length: float
    hub_radius: float
    rim_radius: float

    amount: float

    vane: Vane

    angular_width: float = field(init=False)
    total_radius_range: NDArray[float64] = field(init=False)
    total_area: NDArray[float64] = field(init=False)
    total_volume: NDArray[float64] = field(init=False)

    def __post_init__(self):
        """Computes angular width of cell from number of cells, total cell area and total
        cell volume."""

        self.angular_width = 2*np.pi/self.amount

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
        angular_width = self.angular_width
        thickness = self.vane.thickness

        coeff = 1 - thickness/(angular_width*radius*np.sin(adjacent_angle))

        return coeff

    def area(self, radius_array: NDArray[float64] | None = None) -> NDArray[float64]:
        """Computes area of cell with trapezoid integration over specified array of
        radiuses."""

        if radius_array is None:
            radius_array = self.total_radius_range

        func = 1/2*radius_array**2*self.angular_width
        pure_cell_area = np.trapezoid(func, radius_array, axis=0)

        vane_area = self.vane.area(radius_array)

        area = pure_cell_area - vane_area

        return area

    def volume(self, radius_array: NDArray[float64]) -> NDArray[float64]:
        """Computes volume of cell bounded by given radius values and
        number of partitions."""

        area = self.area(radius_array)

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
        area = self.length*self.angular_width*radius*clutter_coeff

        return area


@dataclass
class Frame(ABC):

    length: float

    @overload
    def equation(self, angle: float) -> float:
        ...

    @overload
    def equation(self, angle: NDArray[float64]) -> NDArray[float64]:
        ...

    @abstractmethod
    def equation(self, angle):
        """Represents frame's equation, returns radius-vector length for given angle"""
        raise

    def sector_area(self, angle_array: NDArray[float64],
                    base_radius: float = 0.0) -> NDArray[float64]:
        """Computes area of sector with trapezeoid integration over provided array of
        angles, base radisu arch area is subtracted from integration results."""

        frame_radius_vector = self.equation(angle_array)

        base_area = 1/2*angle_array*base_radius**2
        shape_area = np.trapezoid(1/2*frame_radius_vector**2, angle_array)

        return shape_area - base_area

    def sector_volume(self, angle_range: NDArray[float64],
                      base_radius: float = 0.0) -> NDArray[float64]:
        """Computes volume of sector in frame with trapezoid integration over given array
        of angles."""

        area = self.sector_area(angle_range, base_radius)

        return area*self.length


@dataclass
class CircularFrame(Frame):

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
class EllipticFrame(Frame):

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
