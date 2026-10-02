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

class BaffledCircularPiston:
    def __init__(
        self,
        n_radial=30,
        n_angular=60
    ):
        self.n_radial = n_radial
        self.n_angular = n_angular

    def pressure(
        self,
        freq,
        velocity,
        piston_area,
        distance=1.0,
        angle=0.0,
        medium=AIR
    ):
        freq = np.asarray(freq, dtype=float)
        velocity = np.asarray(velocity)

        omega = 2 * np.pi * freq
        k = omega / medium.speed_of_sound

        a = np.sqrt(piston_area / np.pi)

        # Discretizing the piston surface into small elements
        dr = a / self.n_radial
        dtheta = 2 * np.pi / self.n_angular


        r = (np.arange(self.n_radial) + 0.5) * dr
        theta = (np.arange(self.n_angular) + 0.5) * dtheta

        R_grid, Theta_grid = np.meshgrid(r, theta, indexing="ij")

        distance_grid = np.sqrt( distance ** 2 + R_grid ** 2 - 2 * distance * R_grid * np.sin(angle) * np.cos(Theta_grid))

        dS = (R_grid * dr * dtheta)

        # Frequency axis added explicitly
        propagation = ( np.exp( -1j * k[:, None, None] * distance_grid[None, :, :] ) / distance_grid[None, :, :])

        integral = np.sum( propagation * dS[None, :, :], axis=(1, 2))

        amplitude_factor = ( 1j * medium.density * omega * velocity / (2 * np.pi))

        return (
            amplitude_factor
            * integral
        )