from __future__ import annotations

from src.diagrams.topology import (
    BlockDiagramRepresentation,
    ParallelBranch,
    Signal,
    SummingInput,
    SummingJunction,
    TransferBlock,
)


def _has_vent(system):
    return hasattr(system.enclosure, "vent_volume_velocity")


def _mechanical_admittance_label(system):
    driver_impedance = r"R_{ms}+j\omega M_{ms}+\frac{1}{j\omega C_{ms}}"
    if system.enclosure is None:
        return rf"$\frac{{1}}{{{driver_impedance}}}$"

    if _has_vent(system):
        box_load = r"S_d^2 Z_a"
    else:
        box_load = r"\frac{S_d^2}{j\omega C_{ab}}"
    return rf"$\frac{{1}}{{{driver_impedance}+{box_load}}}$"


def block_diagram(system, output="displacement", distance=1.0):
    valid_outputs = {"velocity", "displacement", "volume_velocity", "pressure"}
    if output not in valid_outputs:
        raise ValueError(f"output must be one of {sorted(valid_outputs)}")
    if output == "pressure" and system.radiation is None:
        raise ValueError("pressure output requires a radiation model")

    signals = [
        Signal("input_voltage", "$V$", "V"),
        Signal("coil_voltage", "$V-e_{back}$", "V"),
        Signal("current", "$I$", "A"),
        Signal("force", "$F$", "N"),
        Signal("velocity", "$U_d$", "m/s"),
        Signal("back_emf", "$e_{back}$", "V"),
    ]
    forward_blocks = [
        TransferBlock(
            "electrical_admittance",
            r"$\frac{1}{R_e+j\omega L_e}$",
            "coil_voltage",
            "current",
        ),
        TransferBlock("force_factor", "$Bl$", "current", "force"),
        TransferBlock(
            "mechanical_admittance",
            _mechanical_admittance_label(system),
            "force",
            "velocity",
            width=5.0 if _has_vent(system) else None,
        ),
    ]

    output_id = output
    if output == "displacement":
        signals.append(Signal("displacement", "$X_d$", "m"))
        forward_blocks.append(
            TransferBlock("integration", r"$\frac{1}{j\omega}$", "velocity", "displacement")
        )
    elif output in {"volume_velocity", "pressure"}:
        signals.append(Signal("volume_velocity", "$Q_d$", "m³/s"))
        forward_blocks.append(
            TransferBlock("piston_area", "$S_d$", "velocity", "volume_velocity")
        )

    branch = None
    if output == "pressure":
        vented = _has_vent(system)
        driver_output = "driver_pressure" if vented else "pressure"
        signals.append(
            Signal(
                driver_output,
                r"$p_d$" if vented else r"$p(r,\omega)$",
                "Pa",
            )
        )
        forward_blocks.append(
            TransferBlock(
                "monopole_radiation",
                system.radiation.pressure_transfer_label(),
                "volume_velocity",
                driver_output,
            )
        )

        if vented:
            signals += [
                Signal("vent_volume_velocity", "$Q_v$", "m³/s"),
                Signal("vent_pressure", r"$p_v$", "Pa"),
                Signal("pressure", r"$p(r,\omega)$", "Pa"),
            ]
            branch = ParallelBranch(
                tap_signal="volume_velocity",
                blocks=(
                    TransferBlock(
                        "vent_flow",
                        r"$-\frac{Z_{ab}}{Z_{av}}$",
                        "volume_velocity",
                        "vent_volume_velocity",
                    ),
                    TransferBlock(
                        "vent_radiation",
                        system.vent_radiation.pressure_transfer_label(),
                        "vent_volume_velocity",
                        "vent_pressure",
                    ),
                ),
                merge=SummingJunction(
                    "pressure_sum",
                    (SummingInput("driver_pressure", 1), SummingInput("vent_pressure", 1)),
                    "pressure",
                ),
            )

    return BlockDiagramRepresentation(
        signals=tuple(signals),
        forward_blocks=tuple(forward_blocks),
        feedback_block=TransferBlock("back_emf", "$Bl$", "velocity", "back_emf"),
        summing_junction=SummingJunction(
            "voltage_sum",
            (SummingInput("input_voltage", 1), SummingInput("back_emf", -1)),
            "coil_voltage",
        ),
        input_id="input_voltage",
        output_id=output_id,
        branch=branch,
    )