from dataclasses import dataclass
import numpy as np
from src.constants import AIR

@dataclass
class Driver:
    Re: float   # ohm
    Le: float   # H

    Fs: float   # Hz
    Qms: float  # dimensionless
    Qes: float  # dimensionless
    
    Mms: float  # kg
    Cms: float  # m/N
    Sd: float   # m²
    Bl: float   # T*m = N/A

    Xmax: float | None = None  # m

    @property
    def omega_s(self):
        return 2 * np.pi * self.Fs

    @property
    def Rms(self):
        return self.omega_s * self.Mms / self.Qms

    @property
    def Qts(self):
        return self.Qms * self.Qes / (self.Qms + self.Qes)

    @property
    def Fs_from_mass_compliance(self):
        return 1 / (2 * np.pi * np.sqrt(self.Mms * self.Cms))
    
    def Vas(self, medium = AIR):
        return medium.density * medium.speed_of_sound**2 * self.Sd**2 * self.Cms