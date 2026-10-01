from dataclasses import dataclass

@dataclass(frozen=True)
class AcousticMedium:
    density: float          # kg/m³
    speed_of_sound: float   # m/s

AIR = AcousticMedium(
    density = 1.21,
    speed_of_sound = 343.0
)