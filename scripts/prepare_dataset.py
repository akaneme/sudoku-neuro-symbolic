from pathlib import Path

import pandas as pd


RAW_PATH = Path("data/raw/sudoku-3m.csv")
OUTPUT_DIR = Path("data/processed")

TRAIN_SIZE = 20_000
VAL_SIZE = 5_000
TEST_SIZE = 5_000


def main():
    print("Loading dataset...")

    df = pd.read_csv(RAW_PATH)

    print(f"Total puzzles: {len(df):,}")

    # Keep only the columns needed for ML.
    df = df[["puzzle", "solution", "clues", "difficulty"]]

    # Remove duplicate puzzles if any.
    df = df.drop_duplicates(subset="puzzle")

    # Remove rows with missing values.
    df = df.dropna(subset=["puzzle", "solution", "difficulty"])

    # Split by difficulty.
    train_pool = df[df["difficulty"] <= 2.0]
    val_pool = df[
        (df["difficulty"] > 2.0)
        & (df["difficulty"] <= 3.0)
    ]
    test_pool = df[df["difficulty"] > 3.0]

    print("\nAvailable puzzles:")
    print(f"Train pool: {len(train_pool):,}")
    print(f"Validation pool: {len(val_pool):,}")
    print(f"Test pool: {len(test_pool):,}")

    # Sample a manageable subset.
    train = train_pool.sample(
        n=TRAIN_SIZE,
        random_state=42,
    )

    val = val_pool.sample(
        n=VAL_SIZE,
        random_state=42,
    )

    test = test_pool.sample(
        n=TEST_SIZE,
        random_state=42,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False,
    )

    val.to_csv(
        OUTPUT_DIR / "val.csv",
        index=False,
    )

    test.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False,
    )

    print("\nDataset created:")
    print(f"Train:      {len(train):,}")
    print(f"Validation: {len(val):,}")
    print(f"Test:       {len(test):,}")

    print("\nDifficulty ranges:")
    print(
        f"Train:      {train['difficulty'].min():.2f}"
        f" → {train['difficulty'].max():.2f}"
    )
    print(
        f"Validation: {val['difficulty'].min():.2f}"
        f" → {val['difficulty'].max():.2f}"
    )
    print(
        f"Test:       {test['difficulty'].min():.2f}"
        f" → {test['difficulty'].max():.2f}"
    )


if __name__ == "__main__":
    main()