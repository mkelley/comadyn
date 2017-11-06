"""util --- General support functions
=====================================

Methods
-------
mhat
vector_rotate

"""

__all__ = [
    'lb2xyz',
    'mhat',
    'vector_rotate'
]

import numpy as np

def lb2xyz(lam, bet=None):
    """Transform longitude and latitude to a unit vector.

    Parameters
    ----------
    lam : float, array, or Nx2 array
      The longitude(s), or an array of longitudes and
      latitudes. [rad]
    bet : float or array, optional
      The latitude(s). [rad]

    Returns
    -------
    xyz : array or 3xN array
      The unit vectors.

    """
    _lam = np.array(lam).squeeze()
    if bet is None:
        return lb2xyz(_lam[0], _lam[1])

    lamr = np.radians(_lam)
    betr = np.radians(np.array(bet).squeeze())
    return np.array((np.cos(betr) * np.cos(lamr),
                     np.cos(betr) * np.sin(lamr),
                     np.sin(betr)))

def mhat(a, axis=-1):
    """Mangitude and unit vector decomposition.

    Parameters
    ----------
    a : array-like
      An array.
    axis : int, optional
      The axis to decompose.  Default is the last axis.
    
    Returns
    -------
    m : ndarray
      The magnitudes.
    hat : ndarray
      The unit vectors.

    """

    _a  = np.array(a)
    axis = (_a.ndim - 1) if axis is None else axis
    if axis < 0:
        axis += _a.ndim
    m = np.sqrt(np.sum(_a**2, axis))
    hat = np.rollaxis(np.rollaxis(_a, axis) / m, 0, axis + 1)
    return m, hat

def vector_rotate(v, v_from, v_to):
    """Rotate `v` the same way that `v_from` should be rotated to match `v_to`.

    Parameters
    ----------
    v : array-like
      The vector(s) to rotate, shape = (3,) or (N, 3).
    v_from : array-like
      The initial reference vector(s), (3,) or (N, 3).
    v_to : array-like
      The final reference vectors(s), (3,) or (N, 3).

    Result
    ------
    r : ndarray
      The rotated vector(s), (3,) or (N, 3).

    """

    from mskpy.util import mhat

    v = np.array(v)
    v_from = np.array(v_from)
    v_to = np.array(v_to)

    if np.allclose(v_from, v_to):
        return v
    
    # algorithms assume Nx3 arrays
    if v.ndim == 1 and v_from.ndim == 1 and v_to.ndim == 1:
        return_single = True
    else:
        return_single = False

    if v.ndim == 1:
        v = v[np.newaxis]
    if v_from.ndim == 1:
        v_from = v_from[np.newaxis]
    if v_to.ndim == 1:
        v_to = v_to[np.newaxis]

    # we have Nx3 arrays, make sure all N are useable
    N = max([x.shape[0] for x in (v, v_from, v_to)])
    assert v.shape[0] in (1, N), 'v shape must be (3,) or (N, 3), where N matches all other inputs.'
    assert v_from.shape[0] in (1, N), '_fromv shape must be (3,) or (N, 3), where N matches all other inputs.'
    assert v_to.shape[0] in (1, N), 'v_to shape must be (3,) or (N, 3), where N matches all other inputs.'

    # Frame 1, x is the reference point, z is the rotation axis
    if v_from.shape[0] != N:
        x1 = np.tile(v_from, N).reshape((N, 3))
    else:
        x1 = v_from

    z1 = mhat(np.cross(v_from, v_to))[1]
    if z1.shape[0] != N:
        z1 = np.tile(z1, N).reshape((N, 3))
        
    y1 = mhat(np.cross(z1, x1))[1]
    f1 = np.dstack((x1, y1, z1))  # Nx3x3

    # Frame 2: x is the new reference point (axis), z is the same as
    # in Frame 1
    if v_to.shape[0] != N:
        x2 = np.tile(v_to, N).reshape((N, 3))
    else:
        x2 = v_to

    y2 = mhat(np.cross(z1, x2))[1]
    f2 = np.dstack((x2, y2, z1))  # Nx3x3

    # project onto Frame 1, re-express in Frame 2
    r = ((f1 * v[:, :, np.newaxis]).sum(1)[:, np.newaxis] * f2).sum(2)  # Nx3
    return r[0] if return_single else r
