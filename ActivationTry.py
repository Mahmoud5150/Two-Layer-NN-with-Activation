import torch
import torch.nn as nn
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

class simpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        return self.net(x)

model = simpleNN()


loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

for epoch in range(10000):
    y_pred = model(feature)
    loss = loss_fn(y_pred, y_true)
    
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch: {epoch:4d} | loss: {loss.item():.6f}")


torch.save(model.state_dict(), "Model.pth")
print(y_pred.detach())

plt.scatter(range(len(y_true)), y_true.detach(), label="True", color="blue")
plt.scatter(range(len(y_pred)), y_pred.detach(), label="Predicted", color="red")
plt.legend()
plt.show()