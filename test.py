import torch

a = torch.ones(2,3,4)
b = torch.sum(a, dim=-1)
print(b)