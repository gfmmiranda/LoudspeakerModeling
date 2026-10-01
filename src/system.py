from dataclasses import dataclass
import numpy as np

from src.constants import AIR
from src.responses import SystemResponse, SystemCharacteristics

class LoudspeakerSystem:
    def __init__(self, driver, enclosure = None, medium = AIR):
        self.driver = driver
        self.enclosure = enclosure
        self.medium = medium

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
            freq,
            self.medium
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

    def solve(self, f, voltage=1.0):
        omega = 2 * np.pi * np.asarray(f)

        Ze = self.electrical_impedance(f)
        Zm = self.mechanical_impedance(f)

        Zref = self.driver.Bl**2 / Zm
        Zin = Ze + Zref

        I = voltage / Zin
        F = self.driver.Bl * I
        v = F / Zm
        x = v / (1j * omega)

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
            displacement=x,
            back_emf=self.driver.Bl * v
        )
    
    