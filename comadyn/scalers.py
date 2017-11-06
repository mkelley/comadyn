# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""scalers --- Parameter scale factors
======================================

Scalers create parameter scale factors.  They may depend on particle
parameters, such as size, heliocentric distance, etc.

Scalers may be chained together using the `*` operator into a
`CompositeScaler`:

  total_scale = SpeedRh() * SpeedRadius()

Once defined, scalers may be removed from the chain:

  del total_scale[0]  # Removes the `SpeedRh` scaler.

To create the a new scale factor from a `Scaler` or `CompositeScaler`,
use the function call syntax:

  p = ...  # define a new `Particle`
  s = total_scale(p)

  # evaluate heliocentric distance scalers at these distances
  s = total_scale(rh=[1, 2, 3])



Classes
-------
Scaler
CompositeScaler
ProductionRateScaler
PSDScaler

ActiveArea
ConstantFactor
FractalPorosity
NormalActiveArea
PSD_Hanner
PSD_PowerLaw
PSD_RemoveLogBias
QRh
QRhDouble
ScatteredLight
SpeedLimit
SpeedRadius
SpeedRh
SunCone
ThermalEmission
UnityScaler


Exceptions
----------
InvalidScaler
MissingGrainModel


Functions
---------
flux_scaler
mass_calibrate

"""

__all__ = [
    'Scaler',
    'CompositeScaler',
    'ActiveArea',
    'ConstantFactor',
    'FractalPorosity',
    'NormalActiveArea',
    'ParameterWeight',
    'PSD_Hanner',
    'PSD_PowerLaw',
    'PSD_RemoveLogBias',
    'QRh',
    'QRhDouble',
    'ScatteredLight',
    'SpeedLimit',
    'SpeedRadius',
    'SpeedRh',
    'SunCone',
    'ThermalEmission',
    'UnityScaler',
    'flux_scaler'
]

from abc import ABC, abstractmethod
import numpy as np
import astropy.units as u

class InvalidScaler(Exception):
    pass

class MissingGrainModel(Exception):
    pass

class Scaler(ABC):
    """Abstract base class for particle scale factors.

    Notes
    -----
    Particle scale factors are multiplicative.

    """

    def __mul__(self, other):
        return CompositeScaler(self, other)

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __call__(self, p=None, **kwargs):
        """Evaluate the scaler.

        If `p` or `kwargs` do not contain all necessary variables, the
        scaler evaluates to 1.0.

        Parameters
        ----------
        p : Particle, optional
          The scaler variables are chosen from this particle parameter
          set.  If not defined, then variables are chosen from the
          keyword arguments.
        **kwargs
          Scaler variables as keyword arguments.

        Result
        ------
        s : float or ndarray
          The scale factor(s).

        Notes
        -----
        One of `p` or `kwargs` must be defined.  If `p` is not `None`,
        `kwargs` are ignored.

        """
        pass

    @abstractmethod
    def formula(self):
        pass

    def _get(self, keys, p, kwargs):
        """Helper for variable getting in `__call__` methods."""
        source = kwargs if p is None else p
        v = tuple()
        for k in keys:
            v += (kwargs[k],)

        return v

class CompositeScaler(Scaler):
    """Collection of chained scalers.

    To create a `CompositeScaler`, multiply two `Scaler` together::

      total_scale = SpeedRh() * SpeedRadius()

    To remove the `SpeedRh` scale::

      del total_scale.scales[0]

    Raises
    ------
    InvalidScaler

    """

    def __init__(self, lscaler, rscaler):
        self.scalers = []

        if isinstance(lscaler, UnityScaler):
            self = rscaler
        elif isinstance(rscaler, UnityScaler):
            self = lscaler
        else:
            self *= lscale
            self *= rscale

    def __mul__(self, scale):
        if isinstance(scale, UnityScaler):
            return self

        composite = self
        if isinstance(scale, Scaler):
            composite.scales.append(scale)
        elif isinstance(scale, CompositeScaler):
            composite.scales.extend(scale.scales)
        else:
            raise InvalidScaler
        return composite

    def __str__(self):
        return ' * '.join([str(s) for s in self.scales])

    def formula(self):
        return '(' + ') * ('.join([s.formula() for s in self.scales]) + ')'

    def __call__(self, p=None, **kwargs):
        return np.prod([s.scale(p=p, **kwargs) for s in self.scales])

class ActiveArea(Scaler):
    """Emission from an active area.

    Scale factor is 1 if inside the ejection cone, 0 otherwise.

    Parameters
    ----------
    w : float
      Cone full opening angle. [rad]
    ll : array
      Longitude and latitude of the active area. [rad]
    pole : array, optional
      The pole in Ecliptic coordinates, angular (lambda, beta) or
      rectangular (x, y, z).  The Vernal equinox will be arbitrarily
      defined.
    body_basis : array, optional
      Nx3 array of x, y, and z unit vectors defining the
      planetocentric coordinate system, in Ecliptic rectangular
      coordinates.

    Notes
    -----
    If `pole` is provided, the Vernal equinox and first solstice will
    be arbitrarity derived.

    The vectors in `body_basis` are:
       `body_basis[0]` (x): the Vernal equinox
       `body_basis[1]` (y): the first solstice
       `body_basis[2]` (z): the pole

    Only one of `pole` or `body_basis` may be provided.

    """

    def __init__(self, w, ll, pole=None, body_basis=None):
        from astropy.coordinates import spherical_to_cartesian
        from .generators import Vej
        
        self.w = w
        self.ll = ll
        
        if body_basis is None:
            if pole is None:
                body_basis = np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1)), float)
            else:
                body_basis = Vej.pole2basis(pole)
        self.body_basis = body_basis

        # active area normal vector
        self.normal = spherical_to_cartesian(1.0, ll[1], ll[0])

    @property
    def vernal_eq(self):
        return body_basis[0]

    @property
    def solstice(self):
        return body_basis[1]

    @property
    def pole(self):
        return body_basis[2]

    def __str__(self):
        return 'ActiveArea({}, {}, body_basis={})'.format(
            self.w, self.ll, np.array2string(self.body_basis, separator=','))

    def __call__(self, p=None, **kwargs):
        """Evaluate the scaler.

        Requires ejection direction, `v_ej`; ejection speed, `s_ej`,
        is optional.

        """
        
        from .util import mhat

        try:
            v_ej = self._get(['v_ej'], p, kwargs)
        except KeyError:
            return 1.0

        try:
            s_ej = self._get(['s_ej'], p, kwargs)
        except KeyError:
            v_ej = mhat(v_ej)[1]
            s_ej = 1.0

        assert v_ej.shape[-1] == 3

        cth = np.sum(self.normal * v_ej, -1) / s_ej

        return (cth >= np.cos(self.w / 2)).astype(int)

    def formula():
        return '1 if within active area, else 0'

    __call__.__doc__ += '\n'.join(Scaler.__call__.__doc__.splitlines()[2:])

class ConstantFactor(Scaler):
    """Constant scale factor."""
    def __init__(self, c):
        self.c = c

    def __str__(self):
        return 'ConstantFactor({})'.format(self.c)

    def formula(self):
        return r"$C = {:.3g}$".format(self.c)

    def __call__(self, p=None, **kwargs):
        return self.c

class FractalPorosity(Scaler):
    """Density scale factor based on fractal porosity.

    For the bulk material density `rho0`, minimum grain size `a0`, and
    fractal dimension `D`::

      rho = rho0 * (a / a0)**(D - 3)

    Parameters
    ----------
    D : float
      Fractal dimension.
    a0 : float, optional
      Minimum grian size.  Particles smaller than this will always be
      solid. [μm]

    """

    def __init__(self, D, a0=0.1):
        self.D = D
        self.a0 = a0

    def __str__(self):
        return 'FractalPorosity(D={}, a0={})'.format(self.D, self.a0)

    def formula(self):
        return r"$P = (a / a_0)^{{D-{:.3f}}}$".format(self.a0, self.D)

    def __call__(self, p=None, **kwargs):
        """Evaluate the scaler.

        Requires `radius`.
        
        """

        try:
            radius = self._get(['radius'], p, kwargs)
        except KeyError:
            return 1.0

        return (radius / self.a0)**(self.D - 3.0)
    
    __call__.__doc__ += '\n'.join(Scaler.__call__.__doc__.splitlines()[2:])

class NormalActiveArea(ActiveArea):
    """Emission from an active area with a normal distribution.

    Scale factor is > 0 and < 1 inside the cone, 0 otherwise.

    Parameters
    ----------
    sigma : float
      The width of the distribution. [rad]
    """
    
    def __init__(self, sigma, w, ll, pole=None, body_basis=None):
        self.sigma = sigma
        ActiveArea.__init__(self, w, ll, pole=pole, body_basis=body_basis)

    def __str__(self):
        return 'NormalActiveArea({}, {}, {}, {})'.format(
            self.sig, self.w, self.ll,
            np.array2string(self.body_basis, separator=','))

    def __call__(self, p=None, **kwargs):
        from .util import mhat
        
        try:
            v_ej = self._get(['v_ej'], p, kwargs)
        except KeyError:
            return 1.0

        try:
            s_ej = self._get(['s_ej'], p, kwargs)
        except KeyError:
            v_ej = mhat(v_ej)[1]
            s_ej = 1.0

        assert v_ej.shape[-1] == 3

        th = np.arccos(np.sum(self.normal * v_ej, -1) / s_ej)
        s = (th >= (self.w / 2)).astype(float)
        s *= np.exp(-th**2 / 2 / self.sigma**2)
        s /= np.sqrt(2 * np.pi) * self.sigma
        return s

    __doc__ += '\n'.join(ActiveArea.__doc__.splitlines()[6:])

class ParameterWeight(Scaler):
    """Scale value based on a parameter.

    Parameters
    ----------
    key : string
      The particle parameter key that defines the scale factor, e.g.,
      'age'.

    """

    def __init__(self, key):
        self.key = key

    def __str__(self):
        return 'ParameterWeight({})'.format(self.key)

    def formula(self):
        return "W = {}".format(self.key)

    def __call__(self, p=None, **kwargs):
        try:
            w = self._get([self.key], p, kwargs)
        except KeyError:
            return 1.0

        return w

class PSD_Hanner(Scaler):
    """Hanner modified power-law particle size distribuion.

    n(a) = Np * (1 - a0 / a)**M * (a0 / a)**N

    Parameters
    ----------
    a0 : float
      Minimum grain radius. [μm]
    N : float
      PSD for large grains (`a >> ap`) is `a**-N`.
    M : float, optional
      `ap = a0 * (M + N) / N`.
    ap : float, optional
      Peak grain radius. [μm]
    Np : float, optional
      Number of grains with radius `ap`.

    Note
    ----
    One of `M` or `ap` must be provided.

    """

    def __init__(self, a0, N, M=None, ap=None, Np=1):
        self.a0 = a0
        self.N = N
        self.M = M
        self.ap = ap
        self.Np = 1

        assert (M is None) != (ap is None), 'One and only one of `M` or `ap` may be provided.'
        
        if M is None:
            self.M = (self.ap / self.a0 - 1) * self.N
        else:
            self.ap = self.a0 * (self.M + self.N) / self.N

    def __str__(self):
        return 'PSD_Hanner({}, {}, ap={}, Np={})'.format(
            self.a0, self.N, self.ap, self.Np)

    def formula(self):
        return r"dn/da = {Np:.3g} (1 - {a0:.2g} / a)^M ({a0:.2g} / a)^N".format(
            a0=self.a0, N=self.N, M=self.M, Np=self.Np)

    def __call__(self, p=None, **kwargs):
        """Evaluate the scaler.

        Requires `radius` in μm.

        """

        try:
            radius = self._get(['radius'], p, kwargs)
        except KeyError:
            return 1.0

        return (self.Np * (1 - self.a0 / radius)**self.M
                * (self.a0 / radius)**self.N)

    __call__.__doc__ += '\n'.join(Scaler.__call__.__doc__.splitlines()[2:])

class PSD_PowerLaw(Scaler):
    """Power law particle size distribuion.

    n(a) = N1 * a**N

    Parameters
    ----------
    N : float
      Power-law slope.
    N1 : float, optional
      Number of 1-μm-radius particles.


    """

    def __init__(self, N, Np=1):
        self.N = N
        self.Np = Np

    def __str__(self):
        return 'PSD_PowerLaw({}, Np={})'.format(self.N, self.Np)

    def formula(self):
        return r"$dn/da = {:.3g}\times\,a^{{{:.1f}}}$".format(
            self.Np, self.N)

    def __call__(self, p=None, **kwargs):
        """Evaluate the scaler.

        Requires `radius` in μm.

        """

        try:
            radius = self._get(['radius'], p, kwargs)
        except KeyError:
            return 1.0

        return self.Np * radius**self.N

    __call__.__doc__ += '\n'.join(Scaler.__call__.__doc__.splitlines()[2:])

class PSD_RemoveLogBias(Scaler):
    """Remove the log bias of a simulation.

    For simulations with radius picked from the `Log` generator.

    Parameters
    ----------
    Nt : float, optional
    aminmax : array, optional
      Normalize to `Nt` total particles over the radius range
      `aminmax`.

    """

    _Nt = None
    _aminmax = None

    def __init__(self, Nt=None, aminmax=None):
        self.Nt = Nt
        self.aminmax = aminmax

    def __str__(self):
        return 'PSD_RemoveLogBias(Nt={}, aminmax={})'.format(
            self.Nt, self.aminmax)

    def formula(self):
        return r"dn/da_{{correction}} = {:.3g} a".format(self.N0)

    @property
    def Nt(self):
        return self._Nt

    @Nt.setter
    def Nt(self, n):
        self._Nt = n
        self._update_N0()

    @property
    def aminmax(self):
        return self._aminmax

    @aminmax.setter
    def aminmax(self, amm):
        self._aminmax = amm
        self._update_N0()

    def _update_N0(self):
        if (self.Nt is not None) and (self.aminmax is not None):
            self.N0 = self.Nt / np.log(max(self.aminmax) / min(self.aminmax))
        else:
            self.N0 = 1.0

    def __call__(self, p=None, **kwargs):
        """Evaluate the scaler.

        Requires `radius`.

        """

        try:
            radius = self._get(['radius'], p, kwargs)
        except KeyError:
            return 1.0
        
        return self.N0 * radius
    
    __call__.__doc__ += '\n'.join(Scaler.__call__.__doc__.splitlines()[2:])

class QRh(Scaler):
    """Dust production rate dependence on heliocentric distance using a single power-law.

    Qd \propto rh_i**k

    Parameters
    ----------
    k : float
      Power-law scale factor slope.

    """

    def __init__(self, k):
        self.k = k

    def __str__(self):
        return 'QRh({})'.format(self.k)

    def formula(self):
        return (r"$Q \propto r_h^{{{}}}$").format(self.k)

    def __call__(self, p=None, **kwargs):
        """Evaluate the scaler.

        Requires rh_i.

        """

        try:
            rh = self._get(['rh_i'], p, kwargs)
        except KeyError:
            return 1.0
        
        return rh**self.k

class QRhDouble(Scaler):
    """Production rate dependence on heliocentric distance using a double power-law.

    Qd \propto rh_i**k1 for rh_i < rh0
    Qd \propto rh_i**k2 for rh_i > rh0

    The width of the transition from `k1` to `k2` is parameterized by
    `k12`.  Larger `k12` yields shorter transitions.  Try 100.

    The function is normalized to 1.0 at `rh0`.

    Parameters
    ----------
    k1, k2 : float
      Power-law scale factor slopes.
    k12 : float
      Parameter controlling the width of the transition from `k1` to
      `k2`.
    rh0 : float
      The transition heliocentric distance. [AU]

    """

    def __init__(self, k1, k2, k12, rh0):
        self.k1 = k1
        self.k2 = k2
        self.k12 = k12
        self.rh0 = rh0

    def __str__(self):
        return 'QRhDouble({}, {}, {}, {})'.format(self.k1, self.k2,
                                                  self.k12, self.rh0)

    def formula(self):
        return (r"""$Q \propto r_h^{{{}}}$ for $r_h < {}$ AU
$Q \propto r_h^{{{}}}$ for $r_h > {}$ AU""").format(self.k1, self.rh0,
                                                self.k2, self.rh0)

    def __call__(self, p=None, **kwargs):
        """Evaluate the scaler.

        Requires `rh_i`.

        """
        try:
            rh = self._get(['rh_i'], p, kwargs)
        except KeyError:
            return 1.0
        
        alpha = ((self.k1 - self.k2) / self.k12)
        s = 2**-alpha
        s *= (rh / self.rh0)**self.k2
        s *= (1 + (rh / self.rh0)**self.k12)**alpha

        return s
    
    __call__.__doc__ += '\n'.join(Scaler.__call__.__doc__.splitlines()[2:])

class ScatteredLight(Scaler):
    """Radius-based scaler to simulate light scattering.

    The scale factor is::

      Qsca * sigma * S / rh_f / Delta**2

    where `sigma` is the cross-sectional area of the grain, and `S` is
    the solar flux at 1 au, `rh_f` and `Delta` are in au.  The
    scattering efficiency is::

      Qsca = (2 * pi * a / wave)**4  for a < wave / 2 / pi
      Qsca = 1.0                     for a >= wave / 2 / pi

    Parameters
    ----------
    wave : float
      Wavelength of the light. [μm]
    unit : astropy Unit or string
      The flux density units of the scale factor.

    """

    def __init__(self, wave, unit='W/(m2 um)'):
        self.unit = unit
        self.wave = wave

    def __str__(self):
        return 'ScatteredLight({}, unit={})'.format(self.wave, repr(self.unit))

    def __call__(self, p=None, **kwargs):
        """Evaluate the scaler.

        Requires `radius`, `rh_f`, and `Delta`.

        """
        
        from mskpy.calib import solar_flux
        
        try:
            radius, rh_f, Delta = self._get(
                ('radius', 'rh_f', 'Delta'), p, kwargs)
        except KeyError:
            return 1.0

        radius = np.array(radius)
        if radius.ndim == 0:
            radius = radius[np.newaxis]

        Q = np.ones_like(radius)
        k = self.wave / 2 / np.pi

        i = radius < k
        if any(i):
            Q[i] = (radius[i] / k)**4

        sigma = np.pi * (radius * 1e-9)**2  # km**2
        S = solar_flux(self.wave, unit=self.unit).value  # at 1 AU
        
        return Q * sigma * S / rh_f**2 / Delta**2

    __call__.__doc__ += '\n'.join(Scaler.__call__.__doc__.splitlines()[2:])

class SpeedLimit(Scaler):
    """Limit speed to given values.

    If the particle speed is outside the range [`smin`, `smax`], the
    returned scale factor is 0.0.  1.0, otherwise.

    Parameters
    ----------
    smin : float, optional
      Minimum ejection speed. [km/s]
    smax : float, optional
      Maximum ejection speed. [km/s]
    scales : Scaler or CompositeScaler, optional
      Normalize the speed with `scales` before applying limits.  For
      example, if a simulation was picked using over a range of
      values, then scaled with `SpeedRadius`, set `scales` to use the
      same SpeedRadius to undo the scaling.

    """
    
    def __init__(self, smin=0, smax=np.inf, scalers=None):
        self.smin = smin
        self.smax = smax
        if scalers is None:
            self.scalers = UnityScaler()
        else:
            self.scalers = scalers

    def __str__(self):
        return 'SpeedLimit(smin={}, smax={}, scalers={})'.format(
            self.smin, self.smax, self.scalers)

    def __call__(self, p=None, **kwargs):
        """Evaluate the scaler.

        Requires `s_ej` plus any variable needed for `self.scalers`.

        """

        try:
            s_ej = self._get(['s_ej'], p, kwargs)
        except KeyError:
            return 1.0
        
        s = s_ej / self.scalers.scale(p=p, **kwargs)
        i = (s < self.smin) + (s > self.smax)
        if np.iterable(i):
            scale = np.ones_like(s)
            if any(i):
                scale[i] = 0.0
        else:
            scale = 0.0 if i else 1.0
        
        return scale

    __call__.__doc__ += '\n'.join(Scaler.__call__.__doc__.splitlines()[2:])

class SpeedRadius(Scaler):
    """Speed scale factor based on grain raidus.

    For `a` measured in micrometers::

      scale = (a / a0)**k

    Parameters
    ----------
    k : float, optional
      Power-law exponent.
    a0 : float, optional
      Normalization radius.

    """

    def __init__(self, k=-0.5, a0=1.0):
        self.k = k
        self.a0 = a0

    def __str__(self):
        return 'SpeedRadius(k={}, a0={})'.format(self.k, self.a0)

    def __call__(self, p=None, **kwargs):
        """Evaluate the scaler.

        Requires `radius`.

        """
        
        try:
            radius = self._get(['radius'], p, kwargs)
        except KeyError:
            return 1.0

        return (radius / self.a0)**self.k

    __call__.__doc__ += '\n'.join(Scaler.__call__.__doc__.splitlines()[2:])

class SpeedRh(Scaler):
    """Speed scale factor based on heliocentric distance.

    For `rh_i` measured in au::

      scale = (rh_i / rh0)**k

    Parameters
    ----------
    k : float, optional
      Power-law exponent.
    rh0 : float, optional
      Normalization distance.

    """

    def __init__(self, k=-0.5, rh0=1.0):
        self.k = k
        self.rh0 = rh0

    def __str__(self):
        return 'SpeedRh(k={}, rh0={})'.format(self.k, self.rh0)

    def __call__(self, p=None, **kwargs):
        """Evaluate the scaler.

        Requires `rh_i`.

        """
        
        try:
            rh = self._get(['rh_i'], p, kwargs)
        except KeyError:
            return 1.0

        return (rh / self.rh0)**self.k

    __call__.__doc__ += '\n'.join(Scaler.__call__.__doc__.splitlines()[2:])

class SunCone(Scaler):
    """A cone of emission ejected toward the Sun.

    Parameters
    ----------
    w : float
      Cone full opening angle. [rad]

    """

    def __init__(self, w):
        self.w = w

    def __str__(self):
        return 'SunCone({})'.format(self.w)

    def __call__(self, p=None, **kwargs):
        """Evaluate the scaler.

        Requires `r_i` and `v_ej`; initial distance to the Sun, `d_i`,
        and ejection speed, `s_ej` are optional.

        """

        from .util import mhat
        
        try:
            r_i, v_ej = self._get(('r_i', 'v_ej'), p, kwargs)
        except KeyError:
            return 1.0

        try:
            s_ej = self._get(['s_ej'], p, kwargs)
        except KeyError:
            v_ej = mhat(v_ej)[1]
            s_ej = 1.0

        try:
            d_i = self._get(['d_i'], p, kwargs)
        except KeyError:
            r_i = mhat(r_i)[1]
            d_i = 1.0

        cth = np.sum(-r_i * v_ej, -1) / d_i / s_ej

        return (cth >= np.cos(self.w / 2.0)).astype(int)

    __call__.__doc__ += '\n'.join(Scaler.__call__.__doc__.splitlines()[2:])

class ThermalEmission(Scaler):
    """Radius-based scaler to simulate thermal emission.

    The scale factor is::

      Qem * sigma * B / Delta**2

    where `sigma` is the cross-sectional area of the grain, and `S` is
    the solar flux.  The scattering efficiency is::

      Qem = 2 * pi * a / wave  for a < wave / 2 / pi
      Qem = 1.0                for a >= wave / 2 / pi

    Parameters
    ----------
    wave : float
      Wavelength of the light. [micrometers]
    unit : astropy Unit or string, optional
      The flux density units of the scale factor.
    composition : Composition, optional
      Use this composition, rather than anything specified in the
      simluation.
    require_grain_model : bool, optional
      If `True`, and a grain temperature model cannot be found, throw
      an exception.  If `False`, use a blackbody temperature as a
      fail-safe model.

    """

    def __init__(self, wave, unit='W/(m2 um)', composition=None,
                 require_grain_model=False):
        self.unit = unit
        self.wave = wave
        self.composition = composition
        self.require_grain_model = require_grain_model
        print('ThermalEmission is assuming solid grains at the median rh')

    def __str__(self):
        return (('ThermalEmission({}, unit={}, composition={}, '
                 'require_grain_model={})'
             ).format(self.wave, repr(self.unit), str(self.composition),
                      self.require_grain_model))

    def __call__(self, p=None, composition=None, **kwargs):
        """Evaluate the scaler.

        Requires `radius`, `rh_f`, `Delta`, and `composition`.  If
        `self.composition` is defined, then it takes precedence.

        """
        from mskpy.util import planck
        from . import particle

        gtm_filename = {'amorphouscarbon': 'am-carbon.fits',
                        'amorphousolivine50': 'am-olivine50.fits'}

        if self.composition is None:
            if p is not None:
                composition = p.params['pfunc']['composition'].split('(')[0]
            else:
                composition = str(composition).split('(')[0]
        else:
            composition = str(self.composition).split('(')[0]
        composition = composition.lower().strip()

        try:
            radius, rh_f, Delta = self._get(
                ('radius', 'rh_f', 'Delta'), p, kwargs)
        except KeyError:
            return 1.0
        
        if composition in gtm_filename:
            from dust import readgtm, gtmInterp
            from scipy import interpolate
            from scipy.interpolate import splrep, splev
            gtm = readgtm(gtm_filename[composition])
            T = np.zeros_like(radius)
            rh = np.median(rh_f)

            T_rh = np.zeros_like(gtm[2])
            for i in range(len(gtm[2])):
                T_rh[i] = splev(rh, splrep(gtm[3], gtm[0][0, i]))

            T = splev(radius, splrep(gtm[2], T_rh))
        else:
            if self.require_grain_model:
                raise MissingGrainModel

            T = 278. / np.sqrt(rh_f)
        
        radius = np.array(radius)
        if radius.ndim == 0:
            radius = radius[np.newaxis]

        Q = np.ones_like(radius)
        k = self.wave / 2 / np.pi
        
        i = radius < k
        if any(i):
            Q[i] = radius[i] / k
            
        sigma = np.pi * (radius * 1e-9)**2  # km**2
        B = planck(self.wave, T, unit=u.Unit(self.unit) / u.sr).value
        
        return Q * sigma * B / Delta**2

    __call__.__doc__ += '\n'.join(Scaler.__call__.__doc__.splitlines()[2:])

class UnityScaler(Scaler):
    """Scale factor of 1.0."""
    def __init__(self):
        pass

    def __str__(self):
        return 'UnityScaler()'

    def __call__(self, p=None, **kwargs):
        return 1.0

def flux_scaler(Qd=0, psd='a^-3.5', thermal=24, scattered=-1, log_bias=True):
    """Weight a comet simulation with commonly used scalers.

    Parameters
    ----------
    Qd : float, optional
      Specify `k` in `QRh(k)`.
    psd : string, optional
      Particle size distribution, one of 'ism', 'a^k', or
      'hanner a0 N ap'.
    thermal : float, optional
      Wavelength of the thermal emission.  Set to <= 0 to
      disable. [micrometers]
    scattered : float, optional
      Wavelength of the scattered light.  Set to <= 0 to
      disable. [micrometers]
    log_bias : bool, optional
      If `True`, include `PSD_RemoveLogBias` in the scaler.

    Returns
    -------
    scale : CompositeScaler

    """

    psd = psd.lower().strip()
    if psd == 'ism':
        psd_scaler = PSD_PowerLaw(-3.5)
    elif psd[0] == 'k':
        psd_scaler = PSD_PowerLaw(float(psd[2:]))
    elif psd.startswith('hanner'):
        a0, N, ap = [float(x) for x in psd.split()[1:]]
        psd_scaler = PSD_Hanner(a0, N, ap=ap)
    else:
        psd_scaler = UnityScaler()

    if thermal <= 0:
        therm = UnityScaler()
    else:
        therm = ThermalEmission(thermal)

    if scattered <= 0:
        scat = UnityScaler()
    else:
        scat = ScatteredLight(scattered)

    return QRh(Qd) * psd_scaler * PSD_RemoveLogBias() * therm * scat

def mass_calibrate(Q0, rh0, arange, scaler, params, n=None):
    """Calibrate a simulation given a dust production rate.

    Currently considers `ProductionRateScaler`s and `PSDScaler`s.

    Parameters
    ----------
    Q0 : Quantity
      The dust production rate (mass per time) at `rh0`.
    rh0 : Quantity
      The heliocentric distance for which `Q0` is valid.
    arange : Quantity array
      The coma radius range over which `Q0` is computed.
    scaler : Scaler or CompositeScaler
      The simluation scale factors.
    params : dict
      The parameters of the simulation.
    n : int, optional
      The number of particles in the simulation.  The default is to
      use `params['nparticles']`, but this may not always be desired.

    Returns
    -------
    calib : float
      The calibration factor for the simulation to place simulation
      particles in units of coma particles.

    """

    from scipy.integrate import quad
    from mskpy import getspiceobj, cal2time
    from . import generators as csg

    Q0 = Q0.to(u.kg / u.s)
    rh0 = rh0.to(u.au)
    arange = arange.to(u.um)

    if n is None:
        n = params['nparticles']

    gen = eval('csg.' + params['pfunc']['age'])
    trange_sim = gen.min() / 365.25, gen.max() / 365.25

    gen = eval('csg.' + params['pfunc']['radius'])
    arange_sim = gen.min(), gen.max()

    # search scaler for production rate scalers
    Q = UnityScaler()
    if isinstance(scaler, ProductionRateScaler):
        Q = Q * eval(str(scaler))
    elif isinstance(scaler, CompositeScaler):
        for i in range(len(scaler.scales)):
            if isinstance(scaler.scales[i], ProductionRateScaler):
                Q = Q * eval(str(scaler.scales[i]))

    # search scaler for size frequency
    PSD = UnityScaler()
    if isinstance(scaler, PSDScaler):
        PSD = PSD * eval(str(scaler))
    elif isinstance(scaler, CompositeScaler):
        for i in range(len(scaler.scales)):
            if isinstance(scaler.scales[i], PSDScaler):
                PSD = PSD * eval(str(scaler.scales[i]))

    if params['comet']['kernel'] == 'None':
        kernel = None
    else:
        kernel = params['comet']['kernel']

    comet = getspiceobj(params['comet']['name'], kernel=kernel)
    t0 = cal2time(params['date'])

    def mass(a):
        # a in um, mass in g
        from numpy import pi
        #m = 4/3. * pi * (a * 1e-4)**3
        m = 4/3. * pi * a**3 * 1e-12
        dnda = PSD.scale_a(a)
        return m * dnda

    def production_rate(age):
        # unitless
        r = comet.r(t0 - age * u.yr)
        rh = np.sqrt(np.dot(r, r)) / 1.495978707e8
        return Q.scale_rh(rh)

    # theoretical mass of the simulation
    # raw particles / year
    m_sim = float(n) / ((trange_sim[1] - trange_sim[0]) * u.yr)
    # kg / year
    m_sim *= quad(mass, *arange_sim)[0] * 1e-3 * u.kg
    # kg
    m_sim *= quad(production_rate, *trange_sim)[0] * u.yr
    m_sim = m_sim.decompose()

    # mass fraction of simulation compared to coma
    m_frac = quad(mass, *arange_sim)[0] / quad(mass, *arange.value)[0]

    # coma mass
    m_com = (Q0 * quad(production_rate, *trange_sim)[0] * u.yr).decompose()

    return (m_com * m_frac / m_sim).value
