import pandas as pd
from src.neural import model
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

from src.neural.dataset import SudokuDataset
from src.neural.model import SudokuCNN


TRAIN_PATH = "data/processed/train.csv"
VAL_PATH = "data/processed/val.csv"

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_cells = 0
    correct_cells = 0

    with torch.no_grad():
        for puzzles, solutions in loader:
            puzzles = puzzles.to(device)
            solutions = solutions.to(device)

            outputs = model(puzzles)

            # CrossEntropyLoss expects classes 0-8.
            targets = solutions - 1

            loss = criterion(
                outputs.reshape(-1, 9),
                targets.reshape(-1),
            )

            total_loss += loss.item() * targets.numel()

            predictions = outputs.argmax(dim=-1)

            correct_cells += (
                predictions == targets
            ).sum().item()

            total_cells += targets.numel()

    average_loss = total_loss / total_cells
    accuracy = correct_cells / total_cells

    return average_loss, accuracy


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Load datasets
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    train_dataset = SudokuDataset(train_df)
    val_dataset = SudokuDataset(val_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print(f"Training puzzles:   {len(train_dataset):,}")
    print(f"Validation puzzles: {len(val_dataset):,}")

    # Model
    model = SudokuCNN().to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimiser
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print("\nStarting training...\n")

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0
        total_cells = 0
        correct_cells = 0

        for puzzles, solutions in train_loader:

            puzzles = puzzles.to(device)
            solutions = solutions.to(device)

            optimizer.zero_grad()

            outputs = model(puzzles)

            # Convert digits 1-9 → classes 0-8
            targets = solutions - 1

            loss = criterion(
                outputs.reshape(-1, 9),
                targets.reshape(-1),
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * targets.numel()

            predictions = outputs.argmax(dim=-1)

            correct_cells += (
                predictions == targets
            ).sum().item()

            total_cells += targets.numel()

        train_loss = total_loss / total_cells
        train_accuracy = correct_cells / total_cells

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Cell Acc: {train_accuracy * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Cell Acc: {val_accuracy * 100:.2f}%"
        )

    print("\nTraining complete.")

    Path("models").mkdir(exist_ok=True)

    torch.save(
        model.state_dict(),
        "models/neural_baseline.pt",
    )   

    print("Model saved to models/neural_baseline.pt")

if __name__ == "__main__":
    main()