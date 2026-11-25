from dataclasses import dataclass
import numpy as np

from src.constants import AIR
from src.components.enclosure import SealedBox
from src.components.radiation import MonopoleRadiation, BaffledCircularPiston

from src.response import SystemResponse

class LoudspeakerSystem:
    def __init__(
            self, 
            driver, 
            enclosure = None,
            radiation = None,
            medium = AIR,
            vent_radiation = None
            ):
        
        self.driver = driver
        self.enclosure = enclosure
        self.radiation = radiation
        self.vent_radiation = vent_radiation if vent_radiation is not None else MonopoleRadiation()
        self.medium = medium

    def block_diagram(self, output="displacement", distance=1.0):
        from src.diagrams.diagrams import block_diagram

        return block_diagram(self, output=output, distance=distance)

    @staticmethod
    def _omega(freq):
        return 2 * np.pi * np.asarray(freq)

    def electrical_impedance(self, freq):
        w = self._omega(freq)

        return (
            self.driver.Re
            + 1j * w * self.driver.Le
        )

    def driver_mechanical_impedance(self, freq):
        w = self._omega(freq)

        return (
            self.driver.Rms
            + 1j * w * self.driver.Mms
            + 1 / (1j * w * self.driver.Cms)
        )
    
    def enclosure_mechanical_impedance(self, freq):
        if self.enclosure is None:
            return 0.0

        Za = self.enclosure.acoustic_impedance(
            freq=freq,
            medium=self.medium,
        )

        return self.driver.Sd**2 * Za
    
    def mechanical_impedance(self, freq):
        return (
            self.driver_mechanical_impedance(freq)
            + self.enclosure_mechanical_impedance(freq)
        )
    
    @property
    def _loading_factor(self):
        if self.enclosure is None:
            return 1.0

        if isinstance(self.enclosure, SealedBox):
            alpha = self.driver.Vas(self.medium) / self.enclosure.volume
            return np.sqrt(1 + alpha)

        raise NotImplementedError(
            "Loaded resonance and Q are only defined for free-air and sealed boxes."
        )

    @property
    def resonance_frequency(self):
        return self.driver.Fs * self._loading_factor

    @property
    def effective_compliance(self):
        return self.driver.Cms / self._loading_factor**2

    @property
    def mechanical_q(self):
        return self.driver.Qms * self._loading_factor

    @property
    def electrical_q(self):
        return self.driver.Qes * self._loading_factor

    @property
    def total_q(self):
        return self.driver.Qts * self._loading_factor

    
    def analytical_transfer_function(self, freq):
        if isinstance(self.enclosure, SealedBox):

            omega = self._omega(freq)
            omega_c = 2 * np.pi * self.resonance_frequency
            Qtc = self.total_q

            s = 1j * omega

            return (s**2) / ( s**2 + (omega_c / Qtc) * s + omega_c**2)
        
        else:
            raise NotImplementedError(
                "Analytical transfer function is only implemented for sealed boxes."
            )

    def solve(self, f, voltage=1.0, distance=1.0, angle=0.0):
        omega = 2 * np.pi * np.asarray(f)

        Ze = self.electrical_impedance(f)
        Zm = self.mechanical_impedance(f)

        Zref = self.driver.Bl**2 / Zm
        Zin = Ze + Zref

        I = voltage / Zin
        F = self.driver.Bl * I
        v = F / Zm
        x = v / (1j * omega)

        U = self.driver.Sd * v

        driver_pressure = None
        if isinstance(self.radiation, MonopoleRadiation):
            driver_pressure = self.radiation.pressure(
                freq=f,
                volume_velocity=U,
                distance=distance,
                medium=self.medium
            )

        elif isinstance(self.radiation, BaffledCircularPiston):
            driver_pressure = self.radiation.pressure(
                freq=f,
                velocity=v,
                piston_area=self.driver.Sd,
                distance=distance,
                angle=angle,
                medium=self.medium
            )

        vent_volume_velocity = None
        vent_velocity = None
        vent_pressure = None
        pressure = driver_pressure

        if hasattr(self.enclosure, "vent_volume_velocity"):
            vent_volume_velocity = self.enclosure.vent_volume_velocity(
                freq=f,
                driver_volume_velocity=U,
                medium=self.medium,
            )
            vent_velocity = vent_volume_velocity / self.enclosure.vent_area

            if driver_pressure is not None:
                vent_pressure = self.vent_radiation.pressure(
                    freq=f,
                    volume_velocity=vent_volume_velocity,
                    distance=distance,
                    medium=self.medium,
                )
                pressure = driver_pressure + vent_pressure

        return SystemResponse(
            frequency=f,
            voltage=voltage,
            electrical_impedance=Ze,
            mechanical_impedance=Zm,
            reflected_mechanical_impedance=Zref,
            input_impedance=Zin,
            current=I,
            force=F,
            velocity=v,
            volume_velocity=U,
            displacement=x,
            back_emf=self.driver.Bl * v,
            pressure=pressure,
            driver_pressure=driver_pressure,
            vent_pressure=vent_pressure,
            vent_volume_velocity=vent_volume_velocity,
            vent_velocity=vent_velocity,
        )
    
    