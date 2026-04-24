from dataclasses import dataclass


@dataclass(slots=True)
class ExperimentConfig:
    input_dim: int = 16
    hidden_dim: int = 64
    output_dim: int = 2
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 5
    seed: int = 42
