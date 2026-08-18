import torch

from src.neural.model import SudokuCNN


model = SudokuCNN()

x = torch.randint(
    0,
    10,
    (32, 9, 9),
)

output = model(x)

print("Input shape :", x.shape)
print("Output shape:", output.shape)