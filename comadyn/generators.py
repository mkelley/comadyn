# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""generators --- Parameter generators
======================================

Generators are used to create particle variables.  They may be fixed,
random, or produce a fixed sequence.  There are two classes of
generators.  The first class produces single values, useful for
producing particle ages, sizes, etc.  The second is based on a 3D
geometry, designed for producing ejection velocity vectors.

Generators are Python iterators and can be used with the `next`
built-in function, or can be used in loops.  An additional method
allows creating multiple values:

  g = Generator()
  v = next(g)        # create one new value
  v = g.next(N=100)  # create 100 new values

  # generate values until `g` is exhausted, which may be never.
  for v in g:
    print(v)


Classes
-------
Generator
CosineAngle
Delta
Grid
Log
Normal
Sequence
Uniform
UniformAngle

Vej
Isotropic
UniformLatitude
Sunward

Exceptions
----------
InvalidDistribution

"""

__all__ = [
    'CosineAngle',
    'Delta',
    'Grid',
    'Log',
    'Normal',
    'Sequence',
    'Uniform',
    'UniformAngle',

    'Isotropic',
    'UniformLatitude',
    'Sunward']

from abc import ABC, ABCMeta, abstractmethod
import numpy as np

class InvalidDistribution(Exception):
    pass

class Generator(ABC):
    """Abstract base class for CometSuite particle generators."""
    def __iter__(self):
        return self

    @abstractmethod
    def __min__(self):
        pass

    @abstractmethod
    def __min__(self):
        pass

    def __next__(self):
        return self.next(1)

    @abstractmethod
    def next(self, N=1):
        """The next `N` values.

        Parameters
        ----------
        N : int, optional
          `N > 0`.

        """
        pass

    @abstractmethod
    def reset(self):
        """Reset to initial state.

        Especially useful for generators that have a prescribed
        sequence.

        """
        pass
        
class CosineAngle(Generator):
    """Polar angle variate for a solid angle distrubution proportional to `cos`.

    Picked from a distribution such that the flux through a solid
    angle at theta is proportional to `cos(theta)`.

    Only valid for `theta <= pi / 2`.

    Parameters
    ----------
    x0, x1 : float, optional
      Minimum and maximum values, 0 <= x0 <= x1 <= pi / 2. [radians]

    """

    def __init__(self, x0=0, x1=np.pi / 2):
        assert x0 <= x1
        assert x0 >= 0
        assert x1 <= np.pi / 2
        self.x0 = x0
        self.x1 = x1

    def __str__(self):
        return "CosineAngle(x0={}, x1={})".format(self.x0, self.x1)

    def __min__(self):
        return self.x0

    def __max__(self):
        return self.x1

    def next(self, N=1):
        from numpy.random import rand
        u = rand(N)
        u = np.arccos(np.sqrt((1 - u) * np.cos(self.x0)**2
                              + u * np.cos(self.x1)**2))
        return u[0] if N == 1 else u

    def reset(self):
        # nothing to be done
        pass

class Delta(Generator):
    """A "random" variate pick from the delta function distribution.

    Parameters
    ----------
    x0 : float, optional
      The location of the delta function.
 
    Returns
    -------
    x : float
      Always returns `x0`.

    """

    def __init__(self, x0=0):
        self.x0 = x0

    def __str__(self):
        return "Delta(x0={})".format(self.x0)

    def __min__(self):
        return self.x0

    def __max__(self):
        return self.x0

    def next(self, N=1):
        if N == 1:
            return self.x0
        else:
            return np.repeat(self.x0, N)

    def reset(self):
        # nothing to be done
        pass

class Grid(Generator):
    """Variate picked from a uniform grid.

    Parameters
    ----------
    x0 : float
      The start value of the sequence.
    x1 : float
      The end value of the sequence.
    num : int
      The number of samples to generate.
    endpoint : bool, optional
      If `True`, `x1` is the last sample.
    log : bool, optional
      Set to `True` if `x0`, `x1`, and the spacing are in log space.
    cycle : int or float('inf'), optional
      Cylce over the sequence `cycle` times.  Set to `inf` to
      infintely cycle over the sequence.
    repeat : int, optional
      Repeat each element `repeat` times.

    Returns
    -------
    x : float
      The next sequence value.

    """

    def __init__(self, x0, x1, num, endpoint=True, log=False,
                 cycle=1, repeat=1):
        self.args = (x0, x1, num)
        self.kwargs = dict(endpoint=endpoint, log=log, cycle=cycle,
                           repeat=repeat)
        if log:
            seq = np.logspace(x0, x1, num, endpoint=endpoint)
        else:
            seq = np.linspace(x0, x1, num, endpoint=endpoint)
        self._repeat = repeat
        if repeat > 1:
            self.seq = np.repeat(seq, repeat)
        else:
            self.seq = np.array(seq)
        self.cycle = cycle
        self.i = -1

    def __str__(self):
        return ("Grid({args[0]}, {args[1]}, {args[2]}, endpoint={kwargs[endpoint]}, log={kwargs[log]}, cycle={kwargs[cycle]}, repeat={kwargs[repeat]})"
                .format(args=self.args, kwargs=self.kwargs))

    def __min__(self):
        return min(self.args[:2])

    def __max__(self):
        return max(self.args[:2])

    def __next__(self):
        self.i += 1
        if self.i >= len(self.seq):
            self.i = 0
            self.cycle -= 1
        if self.cycle <= 0:
            raise StopIteration
        return self.seq[self.i]

    def next(self, N=1):
        x = np.array((next(self) for i in range(N)))
        return x[0] if N == 1 else x

    def reset(self):
        self.__init__(self._x0, self._x1, self._num,
                      endpoint=self._endpoint, log=self._log,
                      cycle=self._cycle, repeat=self._repeat)

class Log(Generator):
    """Random variate with the distribution dn/dlog ~ 1.

    Base 10.

    Parameters
    ----------
    x0, x1 : float, optional
      `log10(min_val)` and `log10(max_val)`.

    """

    def __init__(self, x0=0, x1=1):
        self.x0 = x0
        self.x1 = x1

    def __str__(self):
        return "Log(x0={}, x1={})".format(self.x0, self.x1)

    def __min__(self):
        return 10**self.x0

    def __max__(self):
        return 10**self.x1

    def next(self, N=1):
        from numpy.random import rand
        x = np.exp((rand(N) * (self.x1 - self.x0) + self.x0)
                   * 2.3025850929940459)
        return x[0] if N == 1 else x

    def reset(self):
        # nothing to be done
        pass

class Normal(Generator):
    """Normally distributed random variate.

    Parameters
    ----------
    mu : float, optional
      The center of the distribution.
    sigma : float, optional
      The width of the distribution.
    x0, x1 : float, optional
      Minimum and maximum values.  The generator is not efficient for
      `x0` similar to `x1`.

    Returns
    -------
    x : float
      The random variate.

    """

    def __init__(self, mu=0, sigma=1, x0=-float('inf'), x1=float('inf')):
        self.mu = mu
        self.sigma = sigma

        assert x0 < x1
        self.x0 = x0
        self.x1 = x1

    def __str__(self):
        return "Normal(mu={}, sigma={}, x0={}, x1={})".format(
            self.mu, self.sigma, self.x0, self.x1)

    def __min__(self):
        return self.x0

    def __max__(self):
        return self.x1

    def __next__(self):
        from numpy.random import randn

        for i in range(10000):
            u = randn(1)[0] * self.sigma + self.mu
            if (u >= self.x0) and (u <= self.x1):
                break
        else:
            raise ValueError("Variate limits are too restrictive:"
                             " no good values in 10,000 iterations.")
        return u

    def next(self, N=1):
        x = np.array((next(self) for i in range(N)))
        return x[0] if N == 1 else x

    def reset(self):
        # nothing to be done
        pass

class Sequence(Generator):
    """Variate picked from a sequence.

    Parameters
    ----------
    seq : array
      The sequence to iterate over.
    cycle : int or float('inf'), optional
      Cylce over the sequence `cycle` times.  Set to `inf` to
      infintely cycle over the sequence.
    repeat : int, optional
      Repeat each element `repeat` times.

    Returns
    -------
    x : float
      The next sequence value.

    """

    def __init__(self, seq, cycle=1, repeat=1):
        self._seq = seq
        self._repeat = repeat
        if repeat > 1:
            self.seq = np.repeat(seq, repeat)
        else:
            self.seq = np.array(seq)
        self.cycle = cycle
        self.i = -1

    def __str__(self):
        return "Sequence({}, cycle={}, repeat={})".format(
            np.array2string(self._seq, max_line_width=32768, separator=','),
            self.cycle, self._repeat)

    def __min__(self):
        return min(self._seq)

    def __max__(self):
        return max(self._seq)

    def __next__(self):
        self.i += 1
        if self.i >= len(self.seq):
            self.i = 0                
            self.cycle -= 1
        if self.cycle <= 0:
            raise StopIteration
        return self.seq[self.i]
    
    def next(self, N=1):
        x = np.array((next(self) for i in range(N)))
        return x[0] if N == 1 else x

    def reset(self):
        Sequence.__init__(self, self._seq, cycle=self.cycle,
                          repeat=self._repeat)

class Uniform(Generator):
    """Uniformly distributed random variate.

    Parameters
    ----------
    x0, x1 : float, optional
      Minimum and maximum values.

    Returns
    -------
    x : float
      The random variate.

    """

    def __init__(self, x0=0, x1=1):
        assert x0 <= x1
        self.x0 = x0
        self.x1 = x1

    def __str__(self):
        return "Uniform(x0={}, x1={})".format(self.x0, self.x1)

    def __min__(self):
        return self.x0

    def __max__(self):
        return self.x1

    def next(self, N=1):
        from numpy.random import rand
        x = rand(N) * (self.x1 - self.x0) + self.x0
        return x[0] if N == 1 else x

    def reset(self):
        # nothing to be done
        pass

class UniformAngle(Generator):
    """Polar angle variate for a uniform solid angle distrubution.

    Picked from a distribution such that the flux through a solid
    angle at theta is proportional to 1.

    Parameters
    ----------
    x0, x1 : float, optional
      Minimum and maximum values, 0 <= x0 <= pi. [radians]

    Returns
    -------
    x : float
      Random angle. [radians]

    """

    def __init__(self, x0=0, x1=np.pi):
        assert x0 <= x1
        assert 0 <= x0
        assert x1 <= np.pi
        self.x0 = x0
        self.x1 = x1

    def __str__(self):
        return "UniformAngle(x0={}, x1={})".format(self.x0, self.x1)

    def __min__(self):
        return self.x0

    def __max__(self):
        return self.x1

    def next(self, N=1):
        from numpy.random import rand
        u = rand(N)
        u = np.arccos((1 - u) * np.cos(self.x0) + u * np.cos(self.x1))
        return u[0] if N == 1 else u

    def reset(self):
        # nothing to be done
        pass

class Vej(Generator, metaclass=ABCMeta):
    """Abstract base class for ejection velocity generators.

    Parameters
    ----------
    pole : array, optional
      The pole in Ecliptic coordinates, angular (lambda, beta) or
      rectangular (x, y, z).  The Vernal equinox will be arbitrarily
      defined.
    body_basis : array, optional
      Nx3 array of x, y, and z unit vectors defining the
      planetocentric coordinate system, in Ecliptic rectangular
      coordinates.
    w : float, optional
      Full opening angle of the emission. [radians].
    distribution : string, optional
      The kind of distribution to use when `w` is provided:
        'uniformangle': uniformly distributed in solid angle
        'normal': Gaussian distribution with `w` as the FWHM.
    theta_dist, phi_dist : Generator
      Specific polar (`theta`) and azimuthal (`phi`) angle
      distributions.

    Notes
    -----
    If `pole` is provided, the Vernal equinox and first solstice will
    be arbitrarity derived.

    The vectors in `body_basis` are:
       `body_basis[0]` (x): the Vernal equinox
       `body_basis[1]` (y): the first solstice
       `body_basis[2]` (z): the pole

    Only one of `pole` or `body_basis` may be provided.

    If `w` is provided, `theta_dist` and `phi_dist` are ignored.

    `min()` and `max()` operators return the respective limits from
    `phi_dist` and `theta_dist`.

    Attributes
    ----------
    theta_dist : Polar angle generator
    phi_dist : Azimuthal angle generator
    body_basis : 3x3 array, as described above
    pole : polar vector
    vernal_eq : Vernal equinox vector
    solstice : First solstice vector

    Methods
    -------
    axis : xxis of symmetry for the ejection cone.
    next : New ejection velocity direction.
    origin : Planetographic coordinates of a vector.

    Static methods
    -------------
    pole2basis : A helper function for defining `body_basis`.

    ..todo: Incorporate nucleus rotation?

    """

    def __init__(self, pole=None, body_basis=None, w=None,
                 distribution='uniformangle', theta_dist=None, phi_dist=None):
        from .state import State

        if body_basis is None:
            if pole is None:
                body_basis = np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1)), float)
            else:
                body_basis = self.pole2basis(pole)

        self.body_basis = body_basis
        self._w = w
        self._distribution = distribution

        if w is not None:
            self.phi_dist = Uniform(x0=0, x1=2 * np.pi)
            if distribution.lower() == 'uniformangle':
                self.theta_dist = UniformAngle(x0=0, x1=w / 2.0)
            elif distribution.lower() == 'normal':
                self.theta_dist = Normal(x0=0, mu=0, sigma=w / 2.35)
            else:
                raise InvalidDistribution("Only 'UniformAngle' and 'Normal'"
                                          " are implemented for w != None.")
        else:
            if theta_dist is None:
                self.theta_dist = Delta(0)
            else:
                self.theta_dist = theta_dist

            if phi_dist is None:
                self.phi_dist = Delta(0)
            else:
                self.phi_dist = phi_dist

    def __min__(self):
        return min(self.phi_dist), min(self.theta_dist)

    def __max__(self):
        return max(self.phi_dist), max(self.theta_dist)

    @abstractmethod
    def axis(self, init):
        """The axis of symmetry.

        Parameters
        ----------
        init : State
          The state of the parent object (comet) at the time of
          ejection.

        """
        pass

    @property
    def vernal_eq(self):
        return body_basis[0]

    @property
    def solstice(self):
        return body_basis[1]

    @property
    def pole(self):
        return body_basis[2]

    @staticmethod
    def pole2basis(pole, vernal_eq=None):
        """Planetographic basis vectors from pole.

        Parameters
        ----------
        pole : array-like
          Ecliptic coordinates of the pole, may be angular (lambda,
          beta) in degrees, or rectangular (x, y, z).
        vernal_eq : array-like, optional
          Ecliptic coordinates of the Vernal equinox, same format as
          `pole`.  If `None`, then the VE will be generated from `y ×
          pole`, unless `pole == y`, in which case VE will be the
          x-axis.

        Returns
        -------
        body_basis : ndarray
          The basis vectors (3x3 array) for planetographic
          coordinates: `body_basis[0]` (x) is the Vernal eqinox,
          `body_basis[1]` (y) is the first solstice, and
          `body_basis[2]` (z) is the pole.

        """

        from mskpy.util import lb2xyz, mhat

        z = pole if len(pole) == 3 else lb2xyz(pole)
        z = mhat(z)[1]
        
        if vernal_eq is None:
            if np.allclose(z, np.array((0, 1.0, 0))):
                # if the pole is the y-axis, VE should be x-axis
                x = np.array((1.0, 0, 0))
            else:
                # use y-axis × pole
                x = mhat(np.cross((0, 1.0, 0), z))[1]
        else:
            x = vernal_eq if len(vernal_eq) == 3 else lb2xyz(vernal_eq)
            c = np.dot(x, z)
            assert np.isclose(c, 0), 'Pole and vernal equinox must be perpendicular to each other, angle is {} rad'.format(np.arccos(c))

        y = mhat(np.cross(z, x))[1]

        return np.vstack((x, y, z))

    def origin(self, v):
        """Planetographic longitude and latitude of the velocity vector.

        Parameters
        ----------
        v : array-like
          The vector to analyze, shape `(3,)` or `(N, 3)`.
        
        Returns
        -------
        lam : float or ndarray
          Longitude.  [rad]
        bet : float or ndarray
          Latitude.  [rad]

        """
        
        p = (self.body_basis * v[:, np.newaxis]).sum(2)
        origin = np.c_[
            np.arctan2(p[:, 1], p[:, 0]),
            np.arctan2(p[:, 2], np.sqrt(p[:, 0]**2 + p[:, 1]**2))
        ]

        return origin[0], origin[1]

    def next(self, init, N=1):
        """New ejection velocity direction(s).

        If the axis of symmetry is [0, 0, 0], the returned velocity
        will also be [0, 0, 0].

        Parameters
        ----------
        init : State or array of States
          The state(s) of the parent object (comet) at the time(s) of
          ejection.  If an array, it must be of length 1 or `N`.
        N : int, optional
          The number of velocities to generate.

        Returns
        -------
        vhat : ndarray
          Ejection velocity direction, shape `(N, 3)`.

        """

        from . import util

        if np.iterable(init):
            assert len(init) in (1, N), "If `init` is an array, it must have length 1 or `N`."

        # choose theta and phi, define radial vector, all w.r.t. axis
        # of symmetry
        theta = np.pi / 2 - self.theta_dist.next(N)
        phi = self.phi_dist.next(N)
        r = np.c_[np.cos(theta) * np.cos(phi),
                  np.cos(theta) * np.sin(phi),
                  np.sin(theta)]

        # define axis of symmetry
        axis = self.axis(init)

        # anywhere axis == [0, 0, 0], v must be [0, 0, 0]
        if axis.ndim == 1:
            if np.all(axis == 0):
                return np.zeros((N, 3))

        i = np.sum(np.abs(axis), 1) == 0
        v = np.empty(r.shape)
        if np.any(i):
            v[i] = 0
            
        # rotate `r` from axis of symmetry coords to Ecliptic coords
        if np.any(~i):
            if axis.ndim == 1:
                v[~i] = util.vector_rotate(r[~i], [0, 0, 1], axis)
            else:
                v[~i] = util.vector_rotate(r[~i], [0, 0, 1], axis[~i])

        return v
    
    def reset(self):
        self.theta_dist.reset()
        self.phi_dist.reset()
       
class Isotropic(Vej):
    def __init__(self):
        Vej.__init__(self, w=2 * np.pi, distribution='UniformAngle')
        self._axis = np.array([1.0, 0.0, 0.0])

    def axis(self, init):
        """The axis of symmetry, arbitrary.

        Parameters
        ----------
        init : State
          The state of the parent object (comet) at the time of
          ejection.

        """
        return self._axis

    def __str__(self):
        return "Isotropic()"

    __doc__ = ["Isotropic emission."]
    doc = Vej.__doc__.splitlines()
    __doc__.extend(doc[1:doc.index('    Parameters')])
    __doc__.extend(doc[doc.index('    Attributes'):])
    __doc__ = '\n'.join(__doc__)
    del doc

class UniformLatitude(Vej):
    def __init__(self, lrange, pole=None, body_basis=None):
        """Uniform emission from a range of latitudes.

        Vectors are uniform in solid angle.

        Parameters
        ----------
        lrange : array-like
          Latitude range, between -pi/2 and pi/2. [radians]
        pole : array, optional
          The pole in Ecliptic coordinates, angular (lambda, beta) or
          rectangular (x, y, z).
        body_basis : array, optional
          Shape `(N, 3)` array of x, y, and z unit vectors defining
          the planetocentric coordinate system, in Ecliptic
          rectangular coordinates.

        """

        phi_dist = Uniform(x0=0, x1=2 * np.pi)
        th = np.pi / 2 - np.array(lrange)  # convert to polar angle
        theta_dist = UniformAngle(x0=min(th), x1=max(th))
        Vej.__init__(self, pole=pole, body_basis=body_basis,
                     phi_dist=phi_dist, theta_dist=theta_dist)

    def axis(self, init):
        """The axis of symmetry is the pole.

        Parameters
        ----------
        init : State
          The state of the parent object (comet) at the time of
          ejection.

        """
        return self.body_basis[2]

    def __str__(self):
        return "UniformLatitude(body_basis={}, theta_dist={}, phi_dist={})".format(
            np.array2string(self.body_basis, separator=','),
            self.theta_dist, self.phi_dist)

    __doc__ = __doc__.splitlines()
    doc = Vej.__doc__.splitlines()
    __doc__.extend(doc[doc.index('    Attributes'):])
    __doc__ = '\n'.join(__doc__)
    del doc

class Sunward(Vej):
    def axis(self, init):
        """The axis of symmetry: the comet-Sun unit vector.

        Parameters
        ----------
        init : State, or array-like
          The state(s) of the parent object (comet) at the time(s) of
          ejection.

        Returns
        -------
        a : array
          Nx3 if `init` is an array.

        """

        from mskpy.util import mhat
        
        if np.iterable(init):
            r = np.array(list((i.r for i in init)))
        else:
            r = init.r
            
        m, hat = mhat(r)
        i = m == 0
        if np.any(i):
            if r.ndim == 1:
                hat = np.zeros(3)
            else:
                hat[i] = 0

        return -hat

    def __str__(self):
        return ("Sunward(body_basis={}, w={}, distribution='{}',"
                " theta_dist={}, phi_dist={})".format(
                    np.array2string(self.body_basis, separator=','),
                    self._w, self._distribution,
                    self.theta_dist, self.phi_dist))

    __doc__ = ["Ejection velocity cone centered on the sunward vector."]
    __doc__.extend(Vej.__doc__.split('\n')[1:])
    __doc__ = '\n'.join(__doc__)
