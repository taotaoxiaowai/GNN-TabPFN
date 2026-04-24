import torch

from dl_research_kit.models import MLPClassifier


def test_mlp_forward_shape() -> None:
    model = MLPClassifier(input_dim=8, hidden_dim=16, output_dim=3)
    x = torch.randn(4, 8)
    y = model(x)
    assert y.shape == (4, 3)
