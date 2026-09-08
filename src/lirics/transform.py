import numpy as np

# Access indices
X = 0
Y = 1

R = 0
PHI = 1


def rotate(x, y, alpha):
    """Perfroms rotation of given (x,y) points around (0,0) by specified angle alpha
    and returns new points coordinates"""

    x_new = x*np.cos(alpha)-y*np.sin(alpha)
    y_new = x*np.sin(alpha)+y*np.cos(alpha)

    return x_new, y_new


def translate(x, y, dr):
    """Performs displacement of (x,y) points by specified displacement vector dr
    and returns transformed coordinates"""

    x_new = x + dr[X]
    y_new = y + dr[Y]

    return x_new, y_new


def xy_to_rphi(x, y):
    """Performs transformation of (x,y) points from Cartesian coordinate system to
    polar coordinate system (r,phi) and returns transformed coordinates"""

    r = np.sqrt(x**2 + y**2)
    phi = np.arctan(y/x)

    return r, phi


def rphi_to_xy(r, phi):
    """Performs transformation of (r,phi) points from polar coordinate system to
    Cartesian coordinate system (x,y) and returns transformed coordinates"""

    x = r*np.cos(phi)
    y = r*np.sin(phi)

    return x, y
