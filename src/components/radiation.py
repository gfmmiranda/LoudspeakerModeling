import numpy as np

from src.constants import AIR


class MonopoleRadiation:
    def pressure_transfer_label(self, distance=1.0):
        return rf"$\frac{{j\omega\rho_0}}{{2\pi r}}e^{{-jkr}},\quad r={distance:g}\,\mathrm{{m}}$"

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