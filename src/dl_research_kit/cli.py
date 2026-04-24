import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import ExperimentConfig
from .engine import Trainer
from .models import MLPClassifier
from .utils import set_seed


def main() -> None:
    cfg = ExperimentConfig()
    set_seed(cfg.seed)

    x = torch.randn(256, cfg.input_dim)
    y = torch.randint(0, cfg.output_dim, (256,))
    loader = DataLoader(TensorDataset(x, y), batch_size=cfg.batch_size, shuffle=True)

    model = MLPClassifier(cfg.input_dim, cfg.hidden_dim, cfg.output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(model, optimizer, criterion)
    history = trainer.fit(loader, epochs=cfg.epochs)
    print(f"Training done. Final loss: {history[-1]:.6f}")


if __name__ == "__main__":
    main()
