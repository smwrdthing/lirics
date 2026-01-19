from __future__ import annotations

from numpy import float64
from numpy.typing import NDArray
from dataclasses import dataclass, field

import numpy as np
from scipy.constants import g as gravity

from lirics import euler


@dataclass
class FlowField:

    time: float
    angle: NDArray[float64]
    radius: NDArray[float64]

    angular_velocity: float
    fluid_density: float

    radial_velocity: NDArray[float64]
    tangent_velocity: NDArray[float64]

    pressure: NDArray[float64]

    _back: int = field(default=0)
    _mid: int = field(default=1)
    _front: int = field(default=2)
    _rim: int = field(default=0)
    _wall: int = field(default=-1)

    def adopt_boundary_parameters(self, cell_flow: euler.FlowField):

        self.tangent_velocity[self._rim, :] = (
            cell_flow.tangent_component[-1, cell_flow._mid] +
            self.angular_velocity*cell_flow.radius[-1, cell_flow._mid])

        self.radial_velocity[self._rim, :] = (
            cell_flow.radial_component[-1, cell_flow._mid])

        # TODO : add pressure attribute to cell flow field
        self.pressure[self._rim, self._back] = cell_flow.pressure[-1, 0]
        self.pressure[self._rim, self._mid] = (
            cell_flow.pressure[-1, cell_flow._mid])
        self.pressure[self._rim, self._front] = cell_flow.pressure[-1, -1]

    def compute_midline_paramters(self, prior_flow):

        # NOTE
        # Maybe use numba's jit to avoid native loop overhead

        for i in range(len(self.tangent_velocity)):
            # loop here
            pass

    def propagate_parameters_to_boundaries(self):
        pass
