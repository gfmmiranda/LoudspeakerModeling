from dataclasses import dataclass
import numpy as np

from src.constants import AIR


@dataclass
class Driver:
    # Electrical parameters
    Re: float   # ohm
    Le: float   # H

    # Resonance / quality factors
    Fs: float   # Hz
    Qms: float  # dimensionless
    Qes: float  # dimensionless

    # Mechanical / electromechanical parameters
    Mms: float  # kg
    Cms: float  # m/N
    Sd: float   # m²
    Bl: float   # T*m = N/A

    # Operating limits
    Xmax: float | None = None  # m
    Pe: float | None = None    # W

    # Optional original datasheet value.
    # Useful for checking consistency with the equivalent parameters.
    Vas_datasheet: float | None = None  # m³


    # ------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------

    @classmethod
    def from_thiele_small(
        cls,
        *,
        Re,
        Le,
        Fs,
        Qms,
        Qes,
        Vas,
        Sd,
        Xmax=None,
        Pe=None,
        medium=AIR,
    ):
        """
        Construct a Driver from a conventional Thiele-Small parameter set.

        Primary inputs:
            Re   : voice-coil DC resistance [ohm]
            Le   : voice-coil inductance [H]
            Fs   : free-air resonance frequency [Hz]
            Qms  : mechanical quality factor
            Qes  : electrical quality factor
            Vas  : equivalent compliance volume [m³]
            Sd   : effective piston area [m²]
            Xmax : maximum linear excursion [m], optional
            Pe   : rated electrical power [W], optional

        Derived:
            Cms, Mms, Rms, Bl
        """

        omega_s = 2 * np.pi * Fs

        # From:
        #
        # Vas = rho * c² * Sd² * Cms
        #

        Cms = (Vas) / (medium.density * medium.speed_of_sound**2 * Sd**2)

        # From:
        #
        # omega_s² = 1 / (Mms * Cms)
        #
        
        Mms = 1 / (omega_s**2 * Cms)

        # From:
        #
        # Qes = omega_s * Mms * Re / Bl²
        #

        Bl = np.sqrt(omega_s * Mms * Re / Qes)

        return cls(
            Re=Re,
            Le=Le,
            Fs=Fs,
            Qms=Qms,
            Qes=Qes,
            Mms=Mms,
            Cms=Cms,
            Sd=Sd,
            Bl=Bl,
            Xmax=Xmax,
            Pe=Pe,
            Vas_datasheet=Vas,
        )


    # ------------------------------------------------------------
    # Derived parameters
    # ------------------------------------------------------------

    @property
    def omega_s(self):
        return 2 * np.pi * self.Fs


    @property
    def Rms(self):
        """
        Mechanical resistance [N*s/m].

        Qms = omega_s * Mms / Rms
        """
        return (self.omega_s * self.Mms / self.Qms)


    @property
    def Qts(self):
        """
        Total free-air quality factor.
        """
        return (self.Qms * self.Qes / (self.Qms + self.Qes))


    @property
    def Fs_from_mass_compliance(self):
        """
        Resonance frequency reconstructed from Mms and Cms.
        """
        return 1 / (2 * np.pi * np.sqrt(self.Mms * self.Cms))


    def Vas(self, medium=AIR):
        """
        Equivalent compliance volume reconstructed from
        Cms and Sd.

        This is the physically consistent Vas used by the model.
        """
        return (medium.density * medium.speed_of_sound**2 * self.Sd**2 * self.Cms)


    @property
    def Vas_error(self):
        """
        Relative difference between the datasheet Vas and the
        Vas reconstructed from Cms and Sd.

        Returns None if no datasheet Vas was supplied.
        """
        if self.Vas_datasheet is None:
            return None

        return (self.Vas() - self.Vas_datasheet) / self.Vas_datasheet