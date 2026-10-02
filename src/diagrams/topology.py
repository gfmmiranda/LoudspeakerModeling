from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Signal:
    id: str
    label: str
    unit: str


@dataclass(frozen=True)
class TransferBlock:
    id: str
    label: str
    input_signal: str
    output_signal: str


@dataclass(frozen=True)
class SummingInput:
    signal: str
    sign: int


@dataclass(frozen=True)
class SummingJunction:
    id: str
    inputs: tuple[SummingInput, ...]
    output_signal: str


@dataclass(frozen=True)
class BlockDiagramRepresentation:
    signals: tuple[Signal, ...]
    forward_blocks: tuple[TransferBlock, ...]
    feedback_block: TransferBlock
    summing_junction: SummingJunction
    input_id: str
    output_id: str

    def signal(self, signal_id: str) -> Signal:
        return next(signal for signal in self.signals if signal.id == signal_id)