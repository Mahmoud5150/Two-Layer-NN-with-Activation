import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

feature = torch.tensor([
    [1,3],
    [5,3],
    [9,5],
    [0,2],
    [4,8]
],dtype=torch.float32)

y_true = torch.tensor([
    [8],
    [9],
    [10],
    [7],
    [15]
],dtype=torch.float32)

w1 =torch.randn(2,4,requires_grad=True)
b1 =torch.randn(4,requires_grad=True)
w2 =torch.randn(4,1,requires_grad=True)
b2 =torch.randn(1,requires_grad=True)

lr = 0.001

for epoch in range(10000):
    hLayer = F.relu(feature @ w1 + b1)
    y_pred = hLayer @ w2 + b2
    loss = ((y_pred-y_true)**2).mean()
    loss.backward()
    with torch.no_grad():
        w1 -= w1.grad * lr
        b1 -= b1.grad * lr
        w2 -= w2.grad * lr
        b2 -= b2.grad * lr
    w1.grad.zero_()
    b1.grad.zero_()
    w2.grad.zero_()
    b2.grad.zero_()
    if epoch % 1000 == 0:
        print(f"Epoch: {epoch:4d} | loss: {loss.item():.6f}")

print(y_pred.detach())

plt.scatter(range(len(y_true)), y_true.detach(), label="True", color="blue")
plt.scatter(range(len(y_pred)), y_pred.detach(), label="Predicted", color="red")
plt.legend()
plt.show()