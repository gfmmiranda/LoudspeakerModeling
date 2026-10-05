import numpy as np
from scipy.special import j1

from src.constants import AIR


class MonopoleRadiation:
    def pressure_transfer_label(self):
        return rf"$\frac{{j\omega\rho_0}}{{2\pi r}}e^{{-jkr}}$"

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

    def pressure_transfer_label(self):
        return (
            r"$\frac{j\rho_0\omega}{2\pi}"
            r"\int_0^a\int_0^{2\pi}"
            r"\frac{e^{-jkR}}{R}"
            r"r'\,d\phi\,dr'$"
        )

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

        dr = a / self.n_radial
        dtheta = 2 * np.pi / self.n_angular

        r = (np.arange(self.n_radial) + 0.5) * dr
        theta = (np.arange(self.n_angular) + 0.5) * dtheta

        R_grid, Theta_grid = np.meshgrid(r, theta, indexing="ij")

        distance_grid = np.sqrt( distance ** 2 + R_grid ** 2 - 2 * distance * R_grid * np.sin(angle) * np.cos(Theta_grid))

        dS = (R_grid * dr * dtheta)

        # Integrand
        propagation = ( np.exp( -1j * k[:, None, None] * distance_grid[None, :, :] ) / distance_grid[None, :, :])

        integral = np.sum( propagation * dS[None, :, :], axis=(1, 2))

        amplitude_factor = ( 1j * medium.density * omega * velocity / (2 * np.pi))

        return (
            amplitude_factor
            * integral
        )
    
    def pressure_field(
        self,
        freq,
        velocity,
        piston_area,
        distances,
        angles,
        medium=AIR,
    ):
        distances = np.asarray(distances)
        angles = np.asarray(angles)

        pressure = np.empty(
            (len(distances), len(angles)),
            dtype=complex,
        )

        for i, distance in enumerate(distances):
            for j, angle in enumerate(angles):
                pressure[i, j] = self.pressure(
                    freq=np.atleast_1d(freq),
                    velocity=np.atleast_1d(velocity),
                    piston_area=piston_area,
                    distance=distance,
                    angle=angle,
                    medium=medium,
                )[0]

        return pressure
    
    def directivity(
        self,
        freq,
        velocity,
        piston_area,
        angles,
        distance=1.0,
        medium=AIR,
    ):
        angles = np.asarray(angles)

        pressure = np.array([
            self.pressure(
                freq=np.atleast_1d(freq),
                velocity=np.atleast_1d(velocity),
                piston_area=piston_area,
                distance=distance,
                angle=angle,
                medium=medium,
            )[0]
            for angle in angles
        ])

        p_on_axis = self.pressure(
            freq=np.atleast_1d(freq),
            velocity=np.atleast_1d(velocity),
            piston_area=piston_area,
            distance=distance,
            angle=0.0,
            medium=medium,
        )[0]

        return pressure / np.abs(p_on_axis)

    def analytical_directivity(
        self,
        freq,
        piston_area,
        angles,
        medium=AIR,
    ):
        angles = np.asarray(angles)

        a = np.sqrt(piston_area / np.pi)
        k = 2 * np.pi * freq / medium.speed_of_sound

        x = k * a * np.sin(angles)

        D = np.ones_like(x, dtype=float)

        mask = np.abs(x) > 1e-12

        D[mask] = (
            2 * j1(x[mask])
            / x[mask]
        )

        return D
    
    def field_coordinates(
        self,
        distances,
        angles,
    ):
        R, Theta = np.meshgrid(
            distances,
            angles,
            indexing="ij",
        )

        x = R * np.sin(Theta)
        z = R * np.cos(Theta)

        return x, z