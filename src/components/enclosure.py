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