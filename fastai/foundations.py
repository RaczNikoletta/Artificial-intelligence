import torch
from torch import tensor

def matmul(a,b):
  ar,ac = a.shape # n_rows * n_cols
  br,bc = b.shape
  assert ac == br
  c = torch.zeros(ar, bc)
  for i in range(ar):
    for j in range(bc):
      for k in range(ac): c[i,j] += a[i,k] * b[k,j]
  return c

m1 = torch.randn(5,28*28)
m2 = torch.randn(784,10)


a = tensor([10.,6,-4])
b = tensor([2.,8,7])
a + b # tensor([12., 14.,  3.])

a < b # tensor([False,  True,  True])
