# LoudspeakerModeling

Electroacoustic modeling of loudspeaker systems using lumped parameters.

The current models support a driver in free air, a driver in a sealed enclosure,
and an optional output-only monopole radiation model.

## Transfer-function diagrams

`LoudspeakerSystem` builds a signal-flow description from the components present
in the model:

```python
diagram = system.block_diagram(output="displacement")
```

The notation follows the usual control-system convention:

- arrows carry signals such as voltage, current, force, velocity, and pressure;
- rectangular blocks contain transfer functions;
- the summing junction subtracts back-EMF from the applied voltage.

Available outputs are `velocity`, `displacement`, `volume_velocity`, and
`pressure`. Pressure requires a radiation model.

Render the representation with Matplotlib:

```python
from src.rendering import render_block_diagram

render_block_diagram(system.block_diagram(output="displacement"))
```

## Architecture

The implementation deliberately has three small layers:

1. `src/topology.py` defines plain data: `Signal`, `TransferBlock`, the summing
	junction, and `BlockDiagramRepresentation`. It contains no loudspeaker
	equations and no drawing code.
2. `src/diagrams.py` translates a `LoudspeakerSystem` into those data objects.
	The driver always contributes electrical, force-factor, mechanical, and
	back-EMF stages. An enclosure changes the mechanical transfer function. A
	radiation model adds the volume-velocity and pressure stages.
3. `src/rendering.py` assigns positions and draws arrows, blocks, and labels. It
	knows nothing about how the numerical solver computes a response.

The representation mirrors the existing solver; it does not evaluate the model
or replace `solve()`. Keeping construction and rendering separate lets another
renderer be added later without changing the physics.

## Tests

Run the diagram tests with:

```shell
python -m unittest discover -s tests -v
```
