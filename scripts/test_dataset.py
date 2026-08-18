import pandas as pd
from torch.utils.data import DataLoader

from src.neural.dataset import SudokuDataset


df = pd.read_csv("data/processed/train.csv")

dataset = SudokuDataset(df)

print("Dataset size:", len(dataset))

puzzle, solution = dataset[0]

print("\nPuzzle:")
print(puzzle)

print("\nSolution:")
print(solution)

print("\nPuzzle shape:", puzzle.shape)
print("Solution shape:", solution.shape)


loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
)

puzzles, solutions = next(iter(loader))

print("\nBatch:")
print("Puzzle batch shape:", puzzles.shape)
print("Solution batch shape:", solutions.shape)