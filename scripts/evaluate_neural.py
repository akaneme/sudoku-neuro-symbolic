import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.neural.dataset import SudokuDataset
from src.neural.model import SudokuCNN


TEST_PATH = "data/processed/test.csv"
MODEL_PATH = "models/neural_baseline.pt"

BATCH_SIZE = 32


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Load test data
    test_df = pd.read_csv(TEST_PATH)

    test_dataset = SudokuDataset(test_df)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Load model
    model = SudokuCNN().to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
        )
    )

    model.eval()

    correct_cells = 0
    total_cells = 0
    correct_blank = 0
    total_blank = 0
    solved_puzzles = 0
    total_puzzles = 0

    total_time = 0.0

    with torch.no_grad():

        for puzzles, solutions in test_loader:

            puzzles = puzzles.to(device)
            solutions = solutions.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            outputs = model(puzzles)

            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            total_time += elapsed

            predictions = outputs.argmax(dim=-1) + 1

            blank = (puzzles == 0)
            correct_blank += ((predictions == solutions) & blank).sum().item()
            total_blank += blank.sum().item()
            
            # Cell accuracy
            correct_cells += (
                predictions == solutions
            ).sum().item()

            total_cells += solutions.numel()

            # Complete puzzle accuracy
            puzzle_correct = (
                predictions == solutions
            ).all(dim=(1, 2))

            solved_puzzles += puzzle_correct.sum().item()
            total_puzzles += puzzles.size(0)

    cell_accuracy = (
        correct_cells / total_cells * 100
    )
    blank_accuracy = correct_blank / total_blank * 100
    avg_clues = (total_cells - total_blank) / total_puzzles

    puzzle_accuracy = (
        solved_puzzles / total_puzzles * 100
    )

    average_time = (
        total_time / total_puzzles * 1000
    )

    print("\n────────────────────────────────────────")
    print("NEURAL MODEL EVALUATION")
    print("────────────────────────────────────────")

    print(f"Puzzles tested          : {total_puzzles:,}")
    print(f"Correct cells           : {correct_cells:,}")
    print(f"Total cells             : {total_cells:,}")
    print(f"Cell accuracy           : {cell_accuracy:.2f}%")
    print(f"Avg clues per puzzle    : {avg_clues:.1f}")
    print(f"Blank-cell accuracy     : {blank_accuracy:.2f}%  (chance = 11.1%)")
    print(f"Completely solved       : {solved_puzzles:,}")
    print(f"Puzzle accuracy         : {puzzle_accuracy:.2f}%")
    print(f"Average inference time  : {average_time:.3f} ms")

    print("────────────────────────────────────────")


if __name__ == "__main__":
    main()
