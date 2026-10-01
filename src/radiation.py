import numpy as np

from src.constants import AIR


class MonopoleRadiation:
    def pressure(
        self,
        freq,
        volume_velocity,
        distance=1.0,
        medium=AIR
    ):
        freq = np.asarray(freq, dtype=float)
        omega = 2 * np.pi * freq

        rho = medium.density
        c = medium.speed_of_sound

        k = omega / c

        return (
            1j
            * omega
            * rho
            * volume_velocity
            / (2 * np.pi * distance)
            * np.exp(-1j * k * distance)
        )