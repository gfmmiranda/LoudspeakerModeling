from dataclasses import dataclass
import numpy as np

from src.constants import AIR

@dataclass
class SealedBox:
    volume: float
    Ql: float | None = None
    Qa: float | None = None

    def acoustic_impedance(
        self,
        freq,
        resonance_frequency,
        medium=AIR,
    ):
        omega = 2 * np.pi * np.asarray(freq)

        Cab = self.volume / ( medium.density * medium.speed_of_sound**2)

        Zc = 1 / (1j * omega * Cab)

        # Ideal box
        if self.Ql is None and self.Qa is None:
            return Zc

        omega_c = 2 * np.pi * resonance_frequency

        # Absorption loss: series resistance
        if self.Qa is None:
            Ra = 0.0
        else:
            Ra = 1 / (omega_c * Cab * self.Qa)

        Z_abs = Ra + Zc

        # No leakage -> just absorption branch
        if self.Ql is None:
            return Z_abs

        # Leakage loss: parallel resistance
        Rl = self.Ql / (omega_c * Cab)

        return 1 / ( 1 / Rl + 1 / Z_abs)