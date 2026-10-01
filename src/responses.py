from dataclasses import dataclass
import numpy as np

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
    back_emf: np.ndarray

@dataclass
class SystemCharacteristics:
    resonance_frequency: float
    effective_compliance: float
    mechanical_q: float
    electrical_q: float
    total_q: float