from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ImpellerCell(ABC):

    hub_radius: float
    rim_radius: float
    length: float
    angular_width: float
    vane_thickness: float

    area: float = field(init=False)
    volume: float = field(init=False)

    @abstractmethod
    def midline_angle(self, radius):
        pass

    def tangents_angle(self, radius):
        pass

    def midline_length(self, radius):
        pass

    def clutter_coefficient(self, radius):
        pass

    def duct_area(self, radius):
        pass


@dataclass
class Housing(ABC):

    length: float

    @abstractmethod
    def profile(self, angle):
        pass


@dataclass
class CylindricalHousing(Housing):

    excentricity: float

    def profile(self, angle):
        pass


@dataclass
class EllipticHousing(Housing):

    major_semiaxis: float
    minor_semiaxis: float

    def profile(self, angle):
        pass
