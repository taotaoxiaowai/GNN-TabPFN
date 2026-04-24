import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def fit(self, train_loader: DataLoader, epochs: int = 5) -> list[float]:
        history: list[float] = []
        self.model.train()
        for _ in range(epochs):
            epoch_loss = 0.0
            n_samples = 0
            for x, y in tqdm(train_loader, leave=False):
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                loss = self.criterion(logits, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                bs = x.size(0)
                epoch_loss += loss.item() * bs
                n_samples += bs

            history.append(epoch_loss / max(n_samples, 1))
        return history
