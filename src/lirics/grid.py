import numpy as np
from numpy.typing import NDArray
from lirics.design import ImpellerCell


type Numeric = float | np.ndarray
type polarCoordinates = tuple[Numeric, Numeric | None]

# In polarCoordinates one coordinate could be omitted sometimes, in those cases
# missing coordinates is defined withtn the context (e.g. from cell midline shape)
#
# Pave buildibg functions are designed to handle angular shifting internally and expect
# start/stop polar coordinates

_R_IDX = 0
_PHI_IDX = 1
NUM_OF_PATH_POINTS: int = 50


def generate(cell: ImpellerCell, shape: tuple[int, int]) -> tuple[NDArray, NDArray]:
    """Generates rectilinear grid in (r,phi) coordinates representing internals of the
    cell. Generated grid is bounded between shiftet cell midlines coincident with ongoing
    and runaway vanes of the impeller, real vane walls profiles are not accounted for"""

    row, col = shape
    xi, eta = np.linspace(0, 1, row), np.linspace(0, 1, col)
    xi, eta = np.meshgrid(xi, eta, indexing="ij")

    r = xi*(cell.rim_radius - cell.hub_radius) + cell.hub_radius
    phi_mid = cell.midline_angle(r[:, 0])

    phi_back = phi_mid - cell.angular_width/2
    phi_front = phi_mid + cell.angular_width/2

    phi = np.transpose((phi_front - phi_back)*eta.T+phi_back)

    return r, phi


def pave_radial_path(
        cell: ImpellerCell,
        start: polarCoordinates,
        stop: polarCoordinates,
        n=NUM_OF_PATH_POINTS) -> tuple[Numeric, Numeric]:
    """Generates (r,phi) points conformal to (shifted) cell midline. Shifting is handled
    internally by means of provided starting point polar coordinates, angular coordinate
    of stopping point is not used."""

    r = np.linspace(start[_R_IDX], stop[_R_IDX], n)
    dphi = start[_PHI_IDX]-cell.midline_angle(start[_R_IDX])  # angular shift
    phi = cell.midline_angle(r) + dphi

    return r, phi


def pave_angular_path(
        start: polarCoordinates,
        stop: polarCoordinates,
        n=NUM_OF_PATH_POINTS) -> tuple[Numeric, Numeric]:
    """Generates (r,phi) points conformal to arch. Arch radius is read from starting
    point, stopping point radius is not used"""

    if start[_PHI_IDX] is None or stop[_PHI_IDX] is None:
        raise ValueError(
            "Angular coordinate of points must be specified for this function")

    # NOTE Pylance complains despite the check above, should figure out how to mitigate.
    #      Ideally we don't want type checks, this is python, after all
    #      (it is alwayas such a major headache to write types when numpy is involved...)
    phi = np.linspace(start[_PHI_IDX], stop[_PHI_IDX], n)  # do not silence!
    r = np.ones_like(phi)*start[_R_IDX]

    return r, phi


def pave_total_path(
        cell: ImpellerCell,
        start: polarCoordinates,
        stop: polarCoordinates,
        n: tuple[int, int, int] = (NUM_OF_PATH_POINTS,
                                   NUM_OF_PATH_POINTS,
                                   NUM_OF_PATH_POINTS)):
    """Generates total integration path needed for interface reconstruction in the cell
    of liquid ring machine with start and stop points in polar coordinates.

    Function is implemented in the general manner, but actual usage should be confined to
    cases with starting point residing on the midline of the cell.

    Intermediate points on the rim of the cell are constructed and handled internally.
    """

    dphi_start = start[_PHI_IDX] - cell.midline_angle(start[_R_IDX])
    dphi_stop = stop[_PHI_IDX] - cell.midline_angle(stop[_R_IDX])

    r_rim = cell.rim_radius
    rim_start = (r_rim, cell.midline_angle(r_rim) + dphi_start)
    rim_stop = (r_rim, cell.midline_angle(r_rim) + dphi_stop)

    up = pave_radial_path(cell, start, rim_start)
    side = pave_angular_path(rim_start, rim_stop)
    down = pave_radial_path(cell, rim_stop, stop)

    # NOTE Pylance comlains, reasons are the same as in function above
    path = np.vstack((up, side, down))

    return path
