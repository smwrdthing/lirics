from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing import overload
from numpy.typing import NDArray

from numpy import float64
import numpy as np

# TODO : type annotations in methods


def circle_slope(angle):
    """Returns slope of circle for specified angle."""
    return -np.cos(angle)/np.sin(angle)


@dataclass
class Vane(ABC):

    start_radius: float
    end_radius: float
    thickness: float

    @abstractmethod
    def equation(self, radius):
        """Represents vane's midline equation, returns angular coordinate that
        corresponds to given radius."""
        raise

    def slope(self, radius, step=0.5e-3):
        """Computes vane's midline slope for given radial coordinate with
        central finite difference approximation of derivative."""

        left = self.equation(radius-step)
        right = self.equation(radius+step)

        return (right-left)/(2*step)

    def length(self, from_radius=None, to_radius=None, partitions=100):
        """Computes length of vane's bounded piece with numerical trapezoid integration
        and specifeid number of partitions. If boundaries aren't provided total length is
        computed."""

        if from_radius is None:
            from_radius = self.start_radius
        if to_radius is None:
            to_radius = self.end_radius

        radius = np.linspace(from_radius, to_radius, partitions)
        slope = self.slope(radius)

        func = np.sqrt(1+radius**2*slope**2)

        length = np.trapezoid(func, radius, axis=0)

        return length

    def area(self, from_radius=None, to_radius=None, partitions=100):
        """Computes approximate area occupied by vane's bounded piece with numerical
        trapezoid integration and specifeid number of partitions. Area is computeda as
        length multiplied by thickness."""

        length = self.length(from_radius, to_radius, partitions)

        return length*self.thickness

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

    def equation(self, radius):
        return np.zeros_like(radius)

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

    def equation(self, radius):
        angle = np.zeros_like(radius)

        after_transition = (radius >= self.transition_radius)
        radius_after_transition = radius[after_transition]

        angle[after_transition] = (
            np.arcsin(self.arch_radius/self.distance_to_arch_center) -
            np.arccos((radius_after_transition**2 + self.distance_to_arch_center**2 -
                       self.arch_radius**2) /
                      (2*radius_after_transition*self.distance_to_arch_center)))

        raise angle

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
        """Computes angular widthof cell from number of cells, total cell area and total
        cell volume."""

        self.angular_width_of_cell = 2*np.pi/self.number_of_cells

        self.total_area_of_cell = self.area_of_cell(
            from_radius=self.hub_radius, to_radius=self.rim_radius)
        self.total_volume_of_cell = self.total_area_of_cell*self.length

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

    def area_of_radial_flow(self, radius):
        """Computes area of radial flow in the cell for given radius"""

        clutter_coeff = self.clutter_coeff(radius)
        area = self.length*self.angular_width_of_cell*radius*clutter_coeff

        return area


@dataclass
class Case(ABC):

    length: float

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

    def equation(self, angle):

        radius_vector = self.excentricity*np.cos(angle) + np.sqrt(
            self.excentricity**2*np.cos(angle) + self.radius**2)

        return radius_vector


@dataclass
class EllipticCase(Case):

    major_semiaxis: float
    minor_semiaxis: float

    def equation(self, angle):

        radius_vector = (
            (np.cos(angle)/self.minor_semiaxis)**2 +
            (np.sin(angle)/self.major_semiaxis)**2)**-0.5

        return radius_vector


class Geometry:

    def __init__(self, impeller: Impeller, case: Case) -> None:
        self.impeller = impeller
        self.case = case

    def area_of_sector(self, angle):
        """Computes area of full sector fromed by cell and corresponding sector in cell's
        external region."""

        angular_width_of_vane = self.impeller.vane.equation(
            self.impeller.rim_radius)
        half_width_of_cell = self.impeller.angular_width_of_cell/2

        area_of_case_sector = self.case.area_of_sector(
            angle + angular_width_of_vane - half_width_of_cell,
            angle + angular_width_of_vane + half_width_of_cell,
            self.impeller.rim_radius)

        total_area_of_sector = self.impeller.total_area_of_cell + area_of_case_sector

        return total_area_of_sector

    def volume_of_sector(self, angle):
        """Computes volume of full sector fromed by cell and corresponding sector in
        cell's external region."""

        area = self.area_of_sector(angle)

        return area*self.case.length
