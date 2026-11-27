from dataclasses import dataclass
import numpy as np

from src.constants import AIR

@dataclass
class SealedBox:
    volume: float

    def acoustic_impedance(
        self,
        freq,
        medium=AIR,
    ):
        # Ideal box
        omega = 2 * np.pi * np.asarray(freq)
        Cab = self.volume / ( medium.density * medium.speed_of_sound**2)
        Zc = 1 / (1j * omega * Cab)

        return Zc


@dataclass
class VentedBox:
    volume: float                                # m³, net internal volume
    vent_radius: float                           # m
    vent_length: float                           # m, physical length
    end_correction_outer: float | None = None    # m, defaults to flanged 8a/(3π)
    end_correction_inner: float | None = None    # m, defaults to 0
    Ql: float | None = 7.0                       # leakage Q at fb, None = lossless
    Qp: float | None = None                      # vent loss Q at fb, None = lossless

    @property
    def vent_area(self):
        return np.pi * self.vent_radius**2

    @property
    def effective_length(self):
        outer = self.end_correction_outer
        if outer is None:
            outer = 8 * self.vent_radius / (3 * np.pi)

        inner = self.end_correction_inner
        if inner is None:
            inner = 0.0

        return self.vent_length + outer + inner

    def vent_acoustic_mass(self, medium=AIR):
        return medium.density * self.effective_length / self.vent_area

    def box_acoustic_compliance(self, medium=AIR):
        return self.volume / (medium.density * medium.speed_of_sound**2)

    def vent_tuning_frequency(self, medium=AIR):
        Mav = self.vent_acoustic_mass(medium)
        Cab = self.box_acoustic_compliance(medium)
        return 1 / (2 * np.pi * np.sqrt(Mav * Cab))

    def vent_acoustic_radiation_resistance(self, freq, medium=AIR):
        omega = 2 * np.pi * np.asarray(freq)
        return medium.density * omega**2 / (2 * np.pi * medium.speed_of_sound)

    def box_acoustic_impedance(self, freq, medium=AIR):
        # Box compliance, with leakage in parallel when Ql is set.
        omega = 2 * np.pi * np.asarray(freq)
        Y = 1j * omega * self.box_acoustic_compliance(medium)

        if self.Ql is not None:
            omega_b = 2 * np.pi * self.vent_tuning_frequency(medium)
            Y = Y + 1 / (self.Ql * omega_b * self.vent_acoustic_mass(medium))

        return 1 / Y

    def vent_acoustic_impedance(self, freq, medium=AIR):
        # Includes the box spring, as the book's Zmv / Sv^2 does.
        omega = 2 * np.pi * np.asarray(freq)
        Mav = self.vent_acoustic_mass(medium)
        Rrv = self.vent_acoustic_radiation_resistance(freq, medium)

        Zav = (
            Rrv
            + 1j * omega * Mav
            + self.box_acoustic_impedance(freq, medium)
        )

        if self.Qp is not None:
            omega_b = 2 * np.pi * self.vent_tuning_frequency(medium)
            Zav = Zav + omega_b * Mav / self.Qp

        return Zav

    def acoustic_impedance(
        self,
        freq,
        medium=AIR,
    ):
        # Box in parallel with the vent as seen from the driver: Zab - Zab^2 / Zav.
        Zab = self.box_acoustic_impedance(freq, medium)
        Zav = self.vent_acoustic_impedance(freq, medium)

        return Zab - Zab**2 / Zav

    def vent_volume_velocity(
        self,
        freq,
        driver_volume_velocity,
        medium=AIR,
    ):
        # Both volume velocities are positive outward; the vent is driven by the rear pressure.
        Zab = self.box_acoustic_impedance(freq, medium)
        Zav = self.vent_acoustic_impedance(freq, medium)

        return -(Zab / Zav) * driver_volume_velocity
