import time

import pandas as pd

from src.backtracking.solver import solve_puzzle


DATASET_PATH = "data/processed/test.csv"
NUM_PUZZLES = 100


def main():
    df = pd.read_csv(
        DATASET_PATH,
        nrows=NUM_PUZZLES,
    )

    results = []

    print("\n╔══════════════════════════════════════╗")
    print("║       BACKTRACKING EVALUATION        ║")
    print("╚══════════════════════════════════════╝")

    for _, row in df.iterrows():
        puzzle = str(row["puzzle"])
        expected = str(row["solution"])
        difficulty = float(row["difficulty"])

        start = time.perf_counter()

        predicted = solve_puzzle(puzzle)

        elapsed = time.perf_counter() - start

        results.append(
            {
                "difficulty": difficulty,
                "correct": predicted == expected,
                "time_ms": elapsed * 1000,
            }
        )

    results_df = pd.DataFrame(results)

    solved = results_df["correct"].sum()
    total = len(results_df)

    print("\n────────────────────────────────────────")
    print("OVERALL RESULTS")
    print("────────────────────────────────────────")

    print(f"Puzzles tested : {total}")
    print(f"Solved         : {solved}")
    print(f"Failed         : {total - solved}")
    print(f"Success rate   : {solved / total * 100:.2f}%")
    print(f"Average time   : {results_df['time_ms'].mean():.3f} ms")
    print(f"Median time    : {results_df['time_ms'].median():.3f} ms")
    print(f"Fastest        : {results_df['time_ms'].min():.3f} ms")
    print(f"Slowest        : {results_df['time_ms'].max():.3f} ms")

    print("\n────────────────────────────────────────")
    print("RESULTS BY DIFFICULTY")
    print("────────────────────────────────────────")

    difficulty_results = (
        results_df
        .groupby(
            pd.cut(
                results_df["difficulty"],
                bins=5,
            )
        )
        .agg(
            puzzles=("correct", "count"),
            accuracy=("correct", "mean"),
            avg_time_ms=("time_ms", "mean"),
        )
    )

    difficulty_results["accuracy"] *= 100

    print(difficulty_results.to_string())

    print("────────────────────────────────────────")


if __name__ == "__main__":
    main()