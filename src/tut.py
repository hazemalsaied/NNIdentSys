import torch
from torch.autograd import Variable
x = torch.Tensor(5, 3).uniform_(-1, 1)
print(x.size())
y = torch.rand(5, 3)
print(x + y)
result = torch.Tensor(5, 3)
torch.add(x, y, out=result)
print(result)
y.add_(x)
print(y)
print(x[:, 1])
print(x[:, 1])
y = torch.randn(5, 10, 15)
print(y.size())
print(y.view(-1, 15).size())

xx = Variable(torch.Tensor([1.0]),  requires_grad=True)
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = Variable(torch.Tensor([1.0]), requires_grad=True)

l = torch.autograd.loss(x_val, y_val)
l.backward()
