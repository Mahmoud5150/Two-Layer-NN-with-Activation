from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn

class simpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2,4),
            nn.ReLU(),
            nn.Linear(4,1)
        )

    def forward(self, x):
        return self.net(x)
    
model = simpleNN()
model.load_state_dict(torch.load("Model.pth"))
model.eval()

app = FastAPI(title="Two layer Neural Network")

class Features(BaseModel):
    feature: list[float]

@app.get("/")
def home():
    return {"message": "Running..."}

@app.post("/predict")
def predict(data: Features):
    x = torch.tensor([data.feature], dtype=torch.float32)
    with torch.no_grad():
        y_pred = model(x)
    return {"prediction": y_pred.item()}
