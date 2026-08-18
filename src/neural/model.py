import torch
import torch.nn as nn


class SudokuCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=10,
            embedding_dim=32,
        )

        self.network = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 9, kernel_size=1),
        )

    def forward(self, x):
        # x: [batch, 9, 9]

        x = self.embedding(x)

        # [batch, 9, 9, 32]
        x = x.permute(0, 3, 1, 2)

        # [batch, 32, 9, 9]
        x = self.network(x)

        # [batch, 9, 9, 9]
        return x