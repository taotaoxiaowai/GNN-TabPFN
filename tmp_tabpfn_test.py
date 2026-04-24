import torch
from examples.tabpfn_adapter import build_frozen_tabpfn

adapter = build_frozen_tabpfn()
print('adapter ok', type(adapter))
x_context = torch.randn(10, 64, device='cpu', requires_grad=True)
y_context = torch.randint(0, 7, (10,), device='cpu')
x_query = torch.randn(5, 64, device='cpu', requires_grad=True)
logits = adapter(x_context, y_context, x_query)
print('logits', logits.shape, logits.dtype, logits.requires_grad)
