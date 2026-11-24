# Neuro-Symbolic Sudoku Solver

A project comparing classical backtracking with a neural Sudoku solver, with the goal of studying whether symbolic constraints can improve neural generalisation on harder, unseen puzzles.

**Status:** backtracking and CNN baselines are done and evaluated. The hybrid solver is the next phase.

## Approaches

- **Backtracking**: exact symbolic baseline
- **CNN**: pure neural baseline (embedding + 4 convolutional layers, predicts a digit for every cell in one pass)
- **Hybrid**: neural predictions guiding symbolic search *(in progress)*

## Dataset

Puzzles are sampled from the Kaggle dataset [3 Million Sudoku Puzzles with Ratings](https://www.kaggle.com/datasets/radcliffe/3-million-sudoku-puzzles-with-ratings), which gives every puzzle a difficulty rating. They are split by difficulty so the test set is harder than anything seen in training (a deliberate distribution shift).

| Split | Difficulty | Puzzles |
|---|---:|---:|
| Train | 0.00 - 2.00 | 20,000 |
| Validation | 2.10 - 3.00 | 5,000 |
| Test | 3.10 - 8.20 | 5,000 |

The raw dataset is not committed because it is too large for GitHub. Download it from Kaggle and save it as `data/raw/sudoku-3m.csv` (see `data/raw/README.md`).

## Results

| Method | Evaluated on | Complete puzzles solved | Cell accuracy | Blank-cell accuracy | Time |
|---|---|---:|---:|---:|---:|
| Backtracking | 100 sampled test puzzles (difficulty ~3.1-5.3) | 100 / 100 | 100% | 100% | median ~100 ms, mean ~595 ms, range 1.6 ms - 24.0 s (CPU) |
| CNN baseline | all 5,000 test puzzles | 0 / 5,000 | 47.57% | 25.10% (chance: 11.1%) | 0.156 ms per puzzle (GPU, batch of 32) |

How to read the metrics:

- **Cell accuracy**: share of all 81 cells predicted correctly, *including the given clues*. Test puzzles have 24.3 clues on average, so about 30% of this number is clues the model can simply copy.
- **Blank-cell accuracy**: share of the initially empty cells predicted correctly. This is the real measure of what the model learned.
- **Complete puzzles solved**: all 81 cells correct.

Backtracking always finds a valid solution but its runtime varies a lot (milliseconds to 24 s). The CNN is fast but does not produce complete valid solutions, which motivates the hybrid approach.

Training setup: 5 epochs, Adam (lr 0.001), batch size 32, cross-entropy loss, 20,000 training puzzles. Training and validation accuracy were close (about 48% vs 47.5%) and the loss was still falling, so the model is under-trained rather than overfit.

### Limitations

- Backtracking was evaluated on a 100-puzzle sample, not the full test set, and the sample does not reach the hardest puzzles (difficulty above ~5.3).
- The CNN was trained and evaluated once; there are no repeated runs or fixed seeds yet.
- The two timings are not directly comparable: different hardware (CPU vs GPU), and the CNN does not return valid solutions.
- An earlier version of the CNN had a tensor-axis bug in its training and evaluation (fixed in commit `ca8ee44`). The numbers above are from the fixed, retrained model.

## Tech Stack

Python · PyTorch · NumPy · Pandas

## Structure

```text
src/
├── backtracking/
│   └── solver.py
└── neural/
    ├── dataset.py
    ├── model.py
    └── train.py

scripts/
├── prepare_dataset.py
├── evaluate.py
├── evaluate_neural.py
└── test_solver.py
```

## Run

Uses [uv](https://docs.astral.sh/uv/). Run everything from the repo root.

```bash
uv sync

# Prepare dataset (after saving the Kaggle file as data/raw/sudoku-3m.csv)
uv run python -m scripts.prepare_dataset

# Evaluate backtracking
uv run python -m scripts.evaluate

# Train neural baseline
uv run python -m src.neural.train

# Evaluate neural baseline
uv run python -m scripts.evaluate_neural
```

## To-do

- Evaluate backtracking on the full 5,000-puzzle test set
- Add a rule-based baseline (random choice among digits that fit row, column and box) to compare against the CNN
- Train the CNN longer, on more data
- Build the hybrid solver: the CNN ranks candidate digits and backtracking verifies them
- Compare neural, backtracking and hybrid on puzzle accuracy, solve time and difficulty buckets
