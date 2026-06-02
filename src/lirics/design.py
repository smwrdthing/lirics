from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np

from lirics.calculus import function_derivative, tabular_derivative

RADIAL_STEP = 1e-3
ARRAY_SIZE = 500


class Cell(ABC):

    def __init__(
            self, radius_bounds: tuple[float, float],
            width: float, span: float, **extra) -> None:
        super().__init__()

        self.radius_bounds = radius_bounds
        self.width = width
        self.span = span

        self.extra = extra

        self._area = None
        self._volume = None

    @abstractmethod
    def midline_angle(self, radius: float | np.ndarray) -> np.ndarray:
        return np.zeros_like(radius)

    def adjacent_angle(
            self, radius: float | np.ndarray, step: float = RADIAL_STEP) -> np.ndarray:

        angle = self.midline_angle(radius)
        derivative = function_derivative(
            self.midline_angle, radius, RADIAL_STEP)

        theta = np.arctan(
            (np.sin(angle) + radius * np.cos(angle) * derivative) /
            (np.cos(angle) - radius * np.sin(angle) * derivative)
        )

        beta = angle + np.pi/2 - theta

        return beta

    def midline_length(self, radius: np.ndarray):

        angle = self.midline_angle(radius)
        derivative = tabular_derivative(angle, radius)

        dxdr = np.cos(angle) - radius * derivative * np.sin(angle)
        dydr = np.sin(angle) + radius * derivative * np.cos(angle)

        l = np.trapezoid(np.sqrt(dxdr**2 + dydr**2), radius)

        return l

    def area(self, radius: np.ndarray | None = None, vane_thickness=0):

        # NOTE : caching

        if radius is None:
            radius = np.linspace(*self.radius_bounds, ARRAY_SIZE)

        A = np.trapezoid(1/2*(self.span*radius**2), radius) - \
            vane_thickness*self.midline_length(radius)

        return A

    def volume(self, radius: np.ndarray | None = None, vane_thickness=0):

        # NOTE : caching

        if radius is None:
            radius = np.linspace(*self.radius_bounds, ARRAY_SIZE)

        V = self.area(radius, vane_thickness)*self.width

        return V


class StraightCell(Cell):

    def midline_angle(self, radius: float | np.ndarray) -> np.ndarray:
        return super().midline_angle(radius)


class ArchedCell(Cell):

    def midline_angle(self, radius):

        transit_radius = self.extra["transit radius"]
        arch_radius = self.extra["arch radius"]
        arch_center_distance = self.extra["arch center distance"]

        post_transit = (radius >= transit_radius)

        angle = np.zeros_like(radius)
        angle += post_transit*(
            np.arcsin(arch_radius / arch_center_distance) -
            np.arccos(
                (radius**2 + arch_center_distance**2 - arch_radius**2) /
                (2*radius*arch_center_distance)
            )
        )

        return angle


class Case(ABC):

    def __init__(self, width, **extra) -> None:
        super().__init__()
        self.width = width
        self.extra = extra

    @abstractmethod
    def profile(self, angle):
        raise

    def area(self, angle: np.ndarray):

        r = self.profile(angle)
        A = np.trapezoid(1/2*r**2, angle)

        return A

    def volume(self, angle: np.ndarray):
        return self.area(angle)*self.width


class CylinderCase(Case):

    def profile(self, angle):
        pass


class EllipticCase(Case):

    def profile(self, angle):
        pass
