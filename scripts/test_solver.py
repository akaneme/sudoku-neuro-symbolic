# test solver checks if ts actually works 
import time

from src.backtracking.solver import solve_puzzle


def print_board(board_string, title):
    print(f"\n{title}")
    print("┌───────┬───────┬───────┐")

    for row in range(9):
        cells = board_string[row * 9:(row + 1) * 9]

        print(
            "│ "
            + " ".join(cells[0:3])
            + " │ "
            + " ".join(cells[3:6])
            + " │ "
            + " ".join(cells[6:9])
            + " │"
        )

        if row in (2, 5):
            print("├───────┼───────┼───────┤")

    print("└───────┴───────┴───────┘")


puzzle = "1..5.37..6.3..8.9......98...1.......8761..........6...........7.8.9.76.47...6.312"

expected = "198543726643278591527619843914735268876192435235486179462351987381927654759864312"

print("\n╔══════════════════════════════════════╗")
print("║        NEURO-SYMBOLIC SUDOKU         ║")
print("║          Backtracking Test           ║")
print("╚══════════════════════════════════════╝")

print_board(puzzle, "Puzzle")

start = time.perf_counter()

actual = solve_puzzle(puzzle)

elapsed = time.perf_counter() - start

print_board(actual, "Solved")

print("\n────────────────────────────────────────")

if actual == expected:
    print("✓ Solution verified")
else:
    print("✗ Solution does not match expected answer")

print(f"✓ Solving time: {elapsed * 1000:.3f} ms")

print("────────────────────────────────────────\n")