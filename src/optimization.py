from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from src.constants import AIR
from src.components.enclosure import VentedBox
from src.components.radiation import MonopoleRadiation
from src.system import LoudspeakerSystem
from src.utils import magnitude_to_db

P_REF = 20e-6

# Phasors are RMS, while Xmax and the vent speed limit are peak values.
PEAK_FROM_RMS = np.sqrt(2)

FEASIBILITY_TOL = 1e-6

F3_DROP_DB = 3.0


@dataclass
class Requirements:
    """Constraints of the design problem, evaluated at every frequency in `freqs` while the
    system is driven at `voltage`."""

    freqs: np.ndarray
    voltage: float                    # V RMS at the source, before any high-pass
    max_f3: float                     # Hz, of the whole system including any high-pass
    max_vent_speed: float = 17.0      # m/s, peak air speed in the vent
    min_vent_radius: float = 0.015    # m
    max_vent_length: float = 0.40     # m, physical length T
    Ql: float = 7.0
    Qp: float | None = None

    def __post_init__(self):
        # The f3 constraint must be checked at max_f3 itself, not only at the next grid point.
        self.freqs = np.union1d(self.freqs, self.max_f3)


@dataclass
class Performance:
    """Response at the drive voltage, per frequency."""

    excursion: np.ndarray      # m, peak
    vent_flow: np.ndarray      # m^3/s, peak
    level: np.ndarray          # dB re the passband (mass-controlled) asymptote


@dataclass
class Design:
    volume: float              # m^3
    fb: float                  # Hz
    vent_radius: float         # m
    vent_length: float         # m, physical length T
    margins: dict = field(repr=False)

    @property
    def feasible(self):
        return is_feasible(self.margins)

    def box(self, req):
        return VentedBox(self.volume, self.vent_radius, self.vent_length, Ql=req.Ql, Qp=req.Qp)


@dataclass
class MaxSPLCurve:
    frequency: np.ndarray
    spl: np.ndarray                      # dB re 20 µPa, lower of the two limits
    spl_excursion_limited: np.ndarray
    spl_voltage_limited: np.ndarray      # thermal limit, or the drive cap if lower
    voltage: np.ndarray                  # V RMS


def make_grid(f_min=10.0, f_max=1000.0, n=300):
    return np.geomspace(f_min, f_max, n)


def vented_system(driver, volume, fb, *, Ql=7.0, Qp=None, highpass=None, medium=AIR):
    # The response depends on (volume, fb) only: Map and Cab do not depend on the vent area at fixed fb.
    radius = 0.03
    area = np.pi * radius**2
    effective_length = medium.speed_of_sound**2 * area / ((2 * np.pi * fb)**2 * volume)

    box = VentedBox(
        volume,
        radius,
        effective_length,
        end_correction_outer=0.0,
        Ql=Ql,
        Qp=Qp,
    )
    return LoudspeakerSystem(driver, box, radiation=MonopoleRadiation(), medium=medium, highpass=highpass)


def vent_length(volume, fb, radius, medium=AIR):
    """Physical length of a flanged vent of the given radius that tunes the box to fb."""
    area = np.pi * radius**2
    effective_length = medium.speed_of_sound**2 * area / ((2 * np.pi * fb)**2 * volume)
    return effective_length - 8 * radius / (3 * np.pi)


def performance(system, req):
    """Peak excursion, peak vent flow and level relative to the unfiltered passband at req.voltage."""
    response = system.solve(req.freqs, voltage=req.voltage)
    driver = system.driver

    # Mass-controlled asymptote of a monopole into half space, including the Le roll-off.
    passband = (
        system.medium.density * driver.Bl * driver.Sd * req.voltage
        / (2 * np.pi * driver.Mms * np.abs(response.electrical_impedance))
    )

    return Performance(
        excursion=PEAK_FROM_RMS * np.abs(response.displacement),
        vent_flow=PEAK_FROM_RMS * np.abs(response.vent_volume_velocity),
        level=magnitude_to_db(response.pressure, passband),
    )


def f3(freqs, level):
    """Lowest frequency above which `level` stays within F3_DROP_DB of the passband."""
    below = np.flatnonzero(level < -F3_DROP_DB)
    if below.size == 0:
        return float(freqs[0])
    i = below[-1]
    if i == len(freqs) - 1:
        return np.nan
    return float(np.interp(-F3_DROP_DB, level[i:i + 2], freqs[i:i + 2]))


def shortest_vent_radius(volume, fb, peak_vent_flow, req, medium=AIR):
    """Smallest radius meeting the radius limit, the speed limit and T >= 0.

    T(a) = c^2 pi a^2 / (wb^2 V) - 8a / (3 pi) increases for a above its root, so this is also
    the radius of the shortest vent."""
    omega_b = 2 * np.pi * fb
    speed_radius = np.sqrt(peak_vent_flow / (np.pi * req.max_vent_speed))
    zero_length_radius = 8 * omega_b**2 * volume / (3 * np.pi**2 * medium.speed_of_sound**2)
    return float(max(req.min_vent_radius, speed_radius, zero_length_radius))


def evaluate(driver, req, volume, fb, highpass=None, medium=AIR):
    """Design at (volume, fb) with the shortest admissible vent, and its constraint margins.

    A constraint holds where its margin is >= 0."""
    system = vented_system(driver, volume, fb, Ql=req.Ql, Qp=req.Qp, highpass=highpass, medium=medium)
    perf = performance(system, req)

    radius = shortest_vent_radius(volume, fb, perf.vent_flow.max(), req, medium)
    length = vent_length(volume, fb, radius, medium)

    margins = {
        "excursion": 1 - perf.excursion / driver.Xmax,
        "f3": perf.level[req.freqs >= req.max_f3] / F3_DROP_DB + 1,
        "vent_length": np.atleast_1d(1 - length / req.max_vent_length),
    }

    return Design(
        volume=float(volume),
        fb=float(fb),
        vent_radius=radius,
        vent_length=float(length),
        margins=margins,
    )


def is_feasible(margins):
    return all(np.min(m) >= -FEASIBILITY_TOL for m in margins.values())


def design_vented_box(
    driver,
    req,
    *,
    highpass=None,
    volume_bounds=(1e-3, 0.2),
    fb_bounds=(10.0, 150.0),
    starts=None,
    medium=AIR,
):
    """Smallest vented box that meets every constraint in `req`, with an optional fixed high-pass
    filter on the drive.

        minimise    V
        over        V, fb       (a_v = shortest_vent_radius(V, fb))
        subject to  peak cone excursion <= Xmax, at every frequency
                    level >= -3 dB re passband, at every frequency >= max_f3
                    T(V, fb, a_v) <= max_vent_length

    The vent speed, minimum radius and T >= 0 hold by construction of a_v. The problem is not
    convex, so SLSQP runs from several (volume, fb) starts and the smallest feasible result is
    kept. Returns None if no start ends feasible.
    """
    scale = np.array([1e-3, 1.0])    # the optimiser works in L and Hz
    bounds = [(volume_bounds[0] / scale[0], volume_bounds[1] / scale[0]), fb_bounds]

    def constraints(z):
        return np.concatenate(list(evaluate(driver, req, *(z * scale), highpass, medium).margins.values()))

    if starts is None:
        starts = [
            (volume, fb)
            for volume in np.geomspace(5e-3, 80e-3, 4)
            for fb in np.array([0.6, 0.9, 1.2]) * req.max_f3
        ]

    best = None
    for start in starts:
        result = minimize(
            lambda z: z[0],
            np.asarray(start) / scale,
            jac=lambda z: np.array([1.0, 0.0]),
            bounds=bounds,
            constraints=[{"type": "ineq", "fun": constraints}],
            method="SLSQP",
            options={"maxiter": 200, "ftol": 1e-10, "eps": 1e-6},
        )

        design = evaluate(driver, req, *(result.x * scale), highpass, medium)
        if design.feasible and (best is None or design.volume < best.volume):
            best = design

    return best


def max_spl_curve(system, freqs, distance=1.0, max_voltage=None):
    """Loudest SPL at each frequency before the cone reaches Xmax or the drive hits the thermal limit
    (or the optional voltage cap, if lower)."""
    driver = system.driver
    response = system.solve(freqs, voltage=1.0, distance=distance)
    # The limits apply at the terminals, so work per terminal volt; a line-level high-pass
    # then leaves the curve unchanged.
    gain = np.abs(response.voltage)
    pressure = np.abs(response.pressure) / gain

    v_excursion = driver.Xmax / (PEAK_FROM_RMS * np.abs(response.displacement) / gain)
    v_limit = np.sqrt(driver.Pe * driver.Re) if driver.Pe is not None else np.inf
    if max_voltage is not None:
        v_limit = min(v_limit, max_voltage)
    voltage = np.minimum(v_excursion, v_limit)

    return MaxSPLCurve(
        frequency=np.asarray(freqs),
        spl=magnitude_to_db(pressure * voltage, P_REF),
        spl_excursion_limited=magnitude_to_db(pressure * v_excursion, P_REF),
        spl_voltage_limited=magnitude_to_db(pressure * v_limit * np.ones_like(pressure), P_REF),
        voltage=voltage,
    )
