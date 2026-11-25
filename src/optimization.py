from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from src.constants import AIR
from src.components.enclosure import SealedBox, VentedBox
from src.components.radiation import MonopoleRadiation
from src.system import LoudspeakerSystem
from src.utils import magnitude_to_db

P_REF = 20e-6

# Phasors are treated as RMS, while Xmax is a peak value.
RMS_FROM_PEAK = 1 / np.sqrt(2)


@dataclass(frozen=True)
class DesignLimits:
    vent_velocity: float = 15.0      # m/s, peak vent air speed on the max-SPL drive curve
    max_vent_length: float = 0.40    # m, physical length T
    min_vent_radius: float = 0.015   # m
    max_voltage: float | None = None # V RMS drive cap; None keeps only the thermal limit


@dataclass
class MaxSPLCurve:
    frequency: np.ndarray
    spl: np.ndarray                      # dB re 20 µPa, lower of the two limits
    spl_excursion_limited: np.ndarray
    spl_voltage_limited: np.ndarray      # thermal limit, or the drive cap if lower
    spl_per_volt: np.ndarray             # small-signal SPL at 1 V RMS
    voltage: np.ndarray                  # V RMS on the max-SPL curve
    vent_volume_velocity: np.ndarray | None  # |Qv| on the max-SPL curve


@dataclass
class VentSizing:
    vent_radius: float
    vent_length: float
    feasible: bool


@dataclass
class Design:
    volume: float
    fb: float
    curve: MaxSPLCurve
    error: float
    f3: float
    vent: VentSizing

    @property
    def feasible(self):
        return self.vent.feasible


def make_grid(f_min=10.0, f_max=1000.0, n=300):
    return np.geomspace(f_min, f_max, n)


def vented_system(driver, volume, fb, *, Ql=7.0, Qp=None, medium=AIR):
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
    return LoudspeakerSystem(driver, box, radiation=MonopoleRadiation(), medium=medium)


def sealed_system(driver, volume, medium=AIR):
    return LoudspeakerSystem(driver, SealedBox(volume), radiation=MonopoleRadiation(), medium=medium)


def max_spl_curve(system, freqs, distance=1.0, max_voltage=None):
    driver = system.driver

    # The model is linear, so one 1 V solve gives every drive level.
    response = system.solve(freqs, voltage=1.0, distance=distance)
    pressure = np.abs(response.pressure)
    excursion = np.abs(response.displacement)

    v_excursion = RMS_FROM_PEAK * driver.Xmax / excursion
    v_limit = np.sqrt(driver.Pe * driver.Re) if driver.Pe is not None else np.inf
    if max_voltage is not None:
        v_limit = min(v_limit, max_voltage)
    voltage = np.minimum(v_excursion, v_limit)

    vent_flow = None
    if response.vent_volume_velocity is not None:
        vent_flow = np.abs(response.vent_volume_velocity) * voltage

    return MaxSPLCurve(
        frequency=np.asarray(freqs),
        spl=magnitude_to_db(pressure * voltage, P_REF),
        spl_excursion_limited=magnitude_to_db(pressure * v_excursion, P_REF),
        spl_voltage_limited=magnitude_to_db(pressure * v_limit * np.ones_like(pressure), P_REF),
        spl_per_volt=magnitude_to_db(pressure, P_REF),
        voltage=voltage,
        vent_volume_velocity=vent_flow,
    )


def f3_frequency(freqs, spl, ref_band=(100.0, 200.0)):
    """Highest frequency below ref_band where spl falls 3 dB under its mean level in ref_band."""
    freqs = np.asarray(freqs)
    spl = np.asarray(spl)
    in_ref = (freqs >= ref_band[0]) & (freqs <= ref_band[1])
    threshold = np.mean(spl[in_ref]) - 3.0

    for i in np.where(freqs < ref_band[0])[0][::-1]:
        if spl[i] < threshold:
            fraction = (threshold - spl[i]) / (spl[i + 1] - spl[i])
            log_f = np.log10(freqs[i]) + fraction * (np.log10(freqs[i + 1]) - np.log10(freqs[i]))
            return float(10**log_f)

    return float(freqs[0])


def spl_error(spl, reference_db, mode="shape"):
    """RMS dB error. 'shape' removes the mean level, 'level' compares directly,
    'shortfall' only counts SPL below the reference."""
    difference = np.asarray(spl) - np.asarray(reference_db)

    if mode == "shape":
        difference = difference - difference.mean()
    elif mode == "shortfall":
        difference = np.minimum(difference, 0.0)
    elif mode != "level":
        raise ValueError("mode must be 'shape', 'level' or 'shortfall'")

    return float(np.sqrt(np.mean(difference**2)))


def size_vent(volume, fb, vent_flow, limits=DesignLimits(), medium=AIR):
    # Qv does not depend on the vent area, so the speed limit gives the minimum area directly.
    area = max(np.max(vent_flow) / limits.vent_velocity, np.pi * limits.min_vent_radius**2)
    radius = np.sqrt(area / np.pi)

    effective_length = medium.speed_of_sound**2 * area / ((2 * np.pi * fb)**2 * volume)
    length = effective_length - 8 * radius / (3 * np.pi)

    return VentSizing(
        vent_radius=float(radius),
        vent_length=float(length),
        feasible=bool(0.0 < length <= limits.max_vent_length),
    )


def evaluate_design(
    driver,
    volume,
    fb,
    *,
    freqs,
    band,
    reference_db,
    mode="shape",
    limits=DesignLimits(),
    Ql=7.0,
    Qp=None,
    distance=1.0,
    medium=AIR,
    f3_ref_band=(100.0, 200.0),
):
    system = vented_system(driver, volume, fb, Ql=Ql, Qp=Qp, medium=medium)
    curve = max_spl_curve(system, freqs, distance, max_voltage=limits.max_voltage)

    in_band = (freqs >= band[0]) & (freqs <= band[1])
    reference = np.broadcast_to(np.asarray(reference_db, dtype=float), freqs.shape)

    return Design(
        volume=volume,
        fb=fb,
        curve=curve,
        error=spl_error(curve.spl[in_band], reference[in_band], mode),
        f3=f3_frequency(freqs, curve.spl_per_volt, f3_ref_band),
        vent=size_vent(volume, fb, curve.vent_volume_velocity, limits, medium),
    )


def best_tuning(driver, volume, *, fb_bounds=(20.0, 80.0), n_grid=41, max_f3=None, **kwargs):
    """Best feasible fb for a fixed volume, or None if no tuning is feasible."""

    def admissible(design):
        return design.feasible and (max_f3 is None or design.f3 <= max_f3)

    fbs = np.geomspace(*fb_bounds, n_grid)
    designs = [evaluate_design(driver, volume, fb, **kwargs) for fb in fbs]
    scores = np.array([d.error if admissible(d) else np.inf for d in designs])

    if not np.isfinite(scores).any():
        return None

    index = int(np.argmin(scores))
    best = designs[index]

    def penalised(fb):
        design = evaluate_design(driver, volume, fb, **kwargs)
        return design.error if admissible(design) else 1e3

    bracket = (fbs[max(index - 1, 0)], fbs[min(index + 1, n_grid - 1)])
    refined = minimize_scalar(penalised, bounds=bracket, method="bounded", options={"xatol": 1e-3})
    candidate = evaluate_design(driver, volume, float(refined.x), **kwargs)

    if admissible(candidate) and candidate.error < best.error:
        best = candidate

    return best


def volume_sweep(driver, volumes, **kwargs):
    return [best_tuning(driver, volume, **kwargs) for volume in volumes]


def error_grid(driver, volumes, fbs, **kwargs):
    error = np.empty((len(volumes), len(fbs)))
    f3 = np.empty((len(volumes), len(fbs)))
    feasible = np.empty((len(volumes), len(fbs)), dtype=bool)

    for i, volume in enumerate(volumes):
        for j, fb in enumerate(fbs):
            design = evaluate_design(driver, volume, fb, **kwargs)
            error[i, j] = design.error
            f3[i, j] = design.f3
            feasible[i, j] = design.feasible

    return error, feasible, f3
