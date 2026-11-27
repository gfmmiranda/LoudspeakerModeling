from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, freqs_zpk


@dataclass
class HighPassFilter:
    """Ideal line-level Butterworth high-pass ahead of the amplifier: it does not load the driver."""

    corner: float    # Hz, -3 dB point
    order: int = 2

    def transfer(self, freq):
        omega = 2 * np.pi * np.asarray(freq, dtype=float)
        z, p, k = butter(self.order, 2 * np.pi * self.corner, btype="high", analog=True, output="zpk")
        return freqs_zpk(z, p, k, worN=omega)[1]
