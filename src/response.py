from dataclasses import dataclass, field
import numpy as np
from src.utils import magnitude_to_db

@dataclass
class SystemResponse:
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
    volume_velocity: np.ndarray
    back_emf: np.ndarray

    pressure: np.ndarray | None = None

    driver_pressure: np.ndarray | None = None
    vent_pressure: np.ndarray | None = None
    vent_volume_velocity: np.ndarray | None = None
    vent_velocity: np.ndarray | None = None

    source_voltage: np.ndarray | None = None   # before the high-pass; `voltage` is at the terminals

    @property
    def spl(self):
        if self.pressure is None:
            return None

        return magnitude_to_db(
            self.pressure,
            reference=20e-6
        )
    
    def normalized_pressure_response_db(
        self,
        reference_frequency
    ):
        if self.pressure is None:
            return None

        idx = np.argmin(np.abs(self.frequency - reference_frequency))
        reference_pressure = np.abs(self.pressure[idx])

        return magnitude_to_db(
            self.pressure,
            reference=reference_pressure
        )