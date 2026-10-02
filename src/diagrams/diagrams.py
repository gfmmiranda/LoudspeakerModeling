from __future__ import annotations

from src.diagrams.topology import (
    BlockDiagramRepresentation,
    Signal,
    SummingInput,
    SummingJunction,
    TransferBlock,
)


def _mechanical_admittance_label(system):
    driver_impedance = r"R_{ms}+j\omega M_{ms}+\frac{1}{j\omega C_{ms}}"
    if system.enclosure is None:
        return rf"$\frac{{1}}{{{driver_impedance}}}$"

    box_compliance = r"\frac{S_d^2}{j\omega C_{ab}}"
    if system.enclosure.Qa is not None:
        box_compliance = rf"S_d^2\left(R_a+\frac{{1}}{{j\omega C_{{ab}}}}\right)"
    if system.enclosure.Ql is not None:
        absorption_branch = box_compliance
        if system.enclosure.Qa is None:
            absorption_branch = r"\frac{S_d^2}{j\omega C_{ab}}"
        box_compliance = rf"\left(S_d^2R_l\right)\parallel\left({absorption_branch}\right)"
    return rf"$\frac{{1}}{{{driver_impedance}+{box_compliance}}}$"


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
        Signal("velocity", "$v$", "m/s"),
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
        ),
    ]

    output_id = output
    if output == "displacement":
        signals.append(Signal("displacement", "$x$", "m"))
        forward_blocks.append(
            TransferBlock("integration", r"$\frac{1}{j\omega}$", "velocity", "displacement")
        )
    elif output in {"volume_velocity", "pressure"}:
        signals.append(Signal("volume_velocity", "$U$", "m³/s"))
        forward_blocks.append(
            TransferBlock("piston_area", "$S_d$", "velocity", "volume_velocity")
        )

    if output == "pressure":
        signals.append(Signal("pressure", r"$p(r,\omega)$", "Pa"))
        forward_blocks.append(
            TransferBlock(
                "monopole_radiation",
                system.radiation.pressure_transfer_label(distance),
                "volume_velocity",
                "pressure",
            )
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
    )