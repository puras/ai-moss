import torch

from demos.pytorch.d10 import compute_accuracy
from demos.pytorch.d4 import NeuralNetwork
from demos.pytorch.d7 import test_loader
from demos.pytorch.d9 import model

torch.save(model.state_dict(), "model.pth")

m = NeuralNetwork(2, 2)
m.load_state_dict(torch.load("model.pth"))

print(compute_accuracy(m, test_loader))