import numpy as np
from numpy.typing import NDArray
from lirics.design import ImpellerCell

NUM_OF_PATH_POINTS = 50


def generate(cell: ImpellerCell, shape: tuple[int, int]) -> tuple[NDArray, NDArray]:

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
        start_raius,
        stop_raius,
        angular_shift=0,
        num_of_points=NUM_OF_PATH_POINTS
):

    r = np.linspace(start_raius, stop_raius, num_of_points)
    phi = cell.midline_angle(r) + angular_shift

    return r, phi


def pave_angular_path(
        radius,
        start_anlge,
        stop_angle,
        num_of_points=NUM_OF_PATH_POINTS):

    phi = np.linspace(start_anlge, stop_angle, num_of_points)
    r = np.ones_like(phi)*radius

    return r, phi


def pave_total_path(cell: ImpellerCell, start_point, stop_points):

    r_start, phi_start = start_point
    r_stop, phi_stop = stop_points

    r_rim, phi_rim = cell.rim_radius, cell.midline_angle(cell.rim_radius)

    dphi_start = phi_start - cell.midline_angle(r_start)
    dphi_stop = phi_stop - cell.midline_angle(r_stop)

    up = pave_radial_path(cell, r_start, r_rim, dphi_start)
    side = pave_angular_path(r_rim, phi_rim+dphi_start, phi_rim+dphi_stop)
    down = pave_radial_path(cell, r_rim, r_stop, dphi_stop)

    path = np.vstack((up, side, down))

    return path
