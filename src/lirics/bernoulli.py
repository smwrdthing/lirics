from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from typing import Literal
from numpy import float64
from numpy.typing import NDArray

from lirics import euler


@dataclass
class FlowField:

    time: float
    angle: NDArray[float64]
    radius: NDArray[float64]

    fluid_density: float

    tangent_velocity: NDArray[float64] = field(init=False)
    radial_velocity: NDArray[float64] = field(init=False)
    pressure: NDArray[float64] = field(init=False)

    # For indexing convenience
    back: Literal[0] = 0
    mid: Literal[1] = 1
    front: Literal[2] = 2

    def compute_flow(self, cell_flow: euler.FlowField, prior_flow: FlowField):

        time_step = self.time - prior_flow.time
        self.tangent_velocity = np.zeros_like(self.radius[:, self.mid])
        self.pressure = np.zeros_like(self.radius[:, self.mid])

        cell_mid = cell_flow.mid

        # TODO : figure this stuff out properly
        #
        # shift_layer_thickness = (
        #     cell_flow.radial_velocity[-1, cell_mid]*time_step)
        # layer_tangent_velocity = (cell_flow.tangent_velocity[-1, cell_mid] +
        #                           cell_flow.translation_velocity)
        # shift_pressure = (
        #     cell_flow.pressure[-1, cell_mid] +
        #     self.fluid_density*layer_tangent_velocity**2 * shift_layer_thickness /
        #     (cell_flow.radius[-1, cell_mid] + shift_layer_thickness/2))
