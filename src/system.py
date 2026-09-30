from dataclasses import dataclass
import numpy as np


@dataclass
class FreeAirResponse:
    frequency: np.ndarray
    voltage: np.ndarray
    electrical_impedance: np.ndarray
    mechanical_impedance: np.ndarray
    reflected_mechanical_impedance: np.ndarray
    input_impedance: np.ndarray
    current: np.ndarray
    force: np.ndarray
    velocity: np.ndarray
    displacement: np.ndarray
    back_emf: np.ndarray


class LoudspeakerSystem:
    def __init__(self, driver):
        self.driver = driver

    @staticmethod
    def _omega(freq):
        return 2 * np.pi * np.asarray(freq)

    def electrical_impedance(self, freq):
        w = self._omega(freq)

        return (
            self.driver.Re
            + 1j * w * self.driver.Le
        )

    def mechanical_impedance(self, freq):
        w = self._omega(freq)

        return (
            self.driver.Rms
            + 1j * w * self.driver.Mms
            + 1 / (1j * w * self.driver.Cms)
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

        return FreeAirResponse(
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