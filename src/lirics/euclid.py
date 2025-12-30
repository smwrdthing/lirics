from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class Geometry:

    length: float

    hub_radisu: float
    rim_radius: float
    transition_radius: float

    number_of_cell: float
    cell_angular_width: float

    rim_tangent_lines_angle: float

    def __post_init__(self):
        self.cell_angular_width = 2*np.pi/self.number_of_cell

        if self.transition_radius is None:  # check how to implement this better
            self.transition_radius = self.hub_radisu
        if self.rim_tangent_lines_angle is None:
            self.rim_tangent_lines_angle = np.pi/2


@dataclass
class GeometrySingle(Geometry):

    excentricity: float
    case_radius: float


@dataclass
class GeometryDoubleEllipse(Geometry):

    major_axis: float
    minor_axis: float
