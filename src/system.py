from dataclasses import dataclass
import numpy as np

from src.constants import AIR
from src.components.enclosure import SealedBox
from src.response import SystemResponse, SystemCharacteristics

class LoudspeakerSystem:
    def __init__(
            self, 
            driver, 
            enclosure = None,
            radiation = None,
            medium = AIR
            ):
        
        self.driver = driver
        self.enclosure = enclosure
        self.radiation = radiation
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

        chars = self.characteristics()

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
    
    def characteristics(self):
        if self.enclosure is None:
            return SystemCharacteristics(
                resonance_frequency=self.driver.Fs,
                effective_compliance=self.driver.Cms,
                mechanical_q=self.driver.Qms,
                electrical_q=self.driver.Qes,
                total_q=self.driver.Qts,
            )

        alpha = self.driver.Vas(self.medium) / self.enclosure.volume
        factor = np.sqrt(1 + alpha)

        return SystemCharacteristics(
            resonance_frequency=self.driver.Fs * factor,
            effective_compliance=self.driver.Cms / (factor**2),
            mechanical_q=self.driver.Qms * factor,
            electrical_q=self.driver.Qes * factor,
            total_q=self.driver.Qts * factor,
        )

    
    def analytical_transfer_function(self, freq):
        if isinstance(self.enclosure, SealedBox):

            chars = self.characteristics()

            omega = self._omega(freq)
            omega_c = 2 * np.pi * chars.resonance_frequency
            Qtc = chars.total_q

            s = 1j * omega

            return (s**2) / ( s**2 + (omega_c / Qtc) * s + omega_c**2)
        
        else:
            raise NotImplementedError(
                "Analytical transfer function is only implemented for sealed boxes."
            )

    def solve(self, f, voltage=1.0, distance=1.0):
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

        pressure = None
        if self.radiation is not None:
            pressure = self.radiation.pressure(
                freq=f,
                volume_velocity=U,
                distance=distance,
                medium=self.medium
            )

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
            pressure=pressure
        )
    
    