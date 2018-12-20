import torch
import torch.nn as nn

class LR(nn.Module):
    def __init__(self, in_size, out_size):
        super(LR, self).__init__()
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, x):
        out = self.linear(x)
        return out

model = LR(1, 1)

x = torch.Tensor([[1.0], [2.0]])
yhat = model(x)
print(yhat)
