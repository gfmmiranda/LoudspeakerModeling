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
    end_correction_inner: float = 0.0            # m
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

        return self.vent_length + outer + self.end_correction_inner

    def acoustic_mass(self, medium=AIR):
        return medium.density * self.effective_length / self.vent_area

    def acoustic_compliance(self, medium=AIR):
        return self.volume / (medium.density * medium.speed_of_sound**2)

    def tuning_frequency(self, medium=AIR):
        return (
            medium.speed_of_sound
            / (2 * np.pi)
            * np.sqrt(self.vent_area / (self.volume * self.effective_length))
        )

    def vent_impedance(self, freq, medium=AIR):
        omega = 2 * np.pi * np.asarray(freq)
        Map = self.acoustic_mass(medium)

        # Acoustic form of the low-frequency baffled-piston radiation resistance.
        Zap = (
            1j * omega * Map
            + medium.density * omega**2 / (2 * np.pi * medium.speed_of_sound)
        )

        if self.Qp is not None:
            omega_b = 2 * np.pi * self.tuning_frequency(medium)
            Zap = Zap + omega_b * Map / self.Qp

        return Zap

    def acoustic_impedance(
        self,
        freq,
        medium=AIR,
    ):
        # Box compliance, leakage and vent are in parallel as seen from the driver.
        omega = 2 * np.pi * np.asarray(freq)
        Cab = self.acoustic_compliance(medium)

        Y = 1j * omega * Cab + 1 / self.vent_impedance(freq, medium)

        if self.Ql is not None:
            omega_b = 2 * np.pi * self.tuning_frequency(medium)
            Ral = self.Ql * omega_b * self.acoustic_mass(medium)
            Y = Y + 1 / Ral

        return 1 / Y

    def vent_volume_velocity(
        self,
        freq,
        driver_volume_velocity,
        medium=AIR,
    ):
        # Both volume velocities are positive outward; the vent is driven by the rear pressure.
        Zload = self.acoustic_impedance(freq, medium)
        Zap = self.vent_impedance(freq, medium)

        return -(Zload / Zap) * driver_volume_velocity
