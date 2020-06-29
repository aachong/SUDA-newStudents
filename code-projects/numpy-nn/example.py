import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

a = np.random.randn(2, 2)
a

a = np.array(([3, 5, 2], [3, 4, 5]))
b = np.array(([2, 4, 1, 2], [3, 4, 5, 4], [3, 3, 6, 7]))
a.shape, b.shape
np.dot(a, b)
np.multiply(a, b)
x = np.mat(([3, 5, 2], [3, 4, 5]))
y = np.mat(b)
x*y

N, Input, H, Output = 64, 10, 100, 5

d_in = np.random.randn(N, Input)
d_out = np.random.randn(N, Output)
w1 = np.random.randn(Input, H)
w2 = np.random.randn(H, Output)

for it in range(300):
    f1 = np.dot(d_in, w1)
    relu = np.maximum(0, f1)
    pred = np.dot(relu, w2)
    loss = np.square(pred-d_out).sum()

    grad_pred = 2.0*(pred-d_out)
    grad_w2 = np.dot()

d = nn.Dropout(0.5)
a = torch.randn(10, 10)
out = d(a)
out

a = torch.randn(4, 4)
b = torch.tensor([[3, 2, 1, 0],
                  [3, 2, 1, 0],
                  [3, 2, 1, 0],
                  [3, 2, 1, 0]])
a.gather(0, b)
a.matmul(b.float())
b.view(-1,1)
