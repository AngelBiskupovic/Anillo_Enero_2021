import torch


a = torch.randn(128, 50, 30)
print(a[:, 50:, :])
