# Neuro-Symbolic Sudoku Solver

A project comparing classical backtracking with a neural Sudoku solver,
with the goal of studying whether symbolic constraints can improve
neural generalisation on harder, unseen puzzles.

## Approaches

- **Backtracking** — symbolic baseline
- **CNN** — pure neural baseline
- **Hybrid** — neural predictions + symbolic constraints *(next phase)*

## Dataset

3M-puzzle Sudoku dataset, split by difficulty to create a distribution
shift:

| Split | Difficulty | Size |
|---|---:|---:|
| Train | ≤ 2.0 | 20,000 |
| Validation | 2.0–3.0 | 5,000 |
| Test | > 3.0 | 5,000 |

## Results

| Model | Accuracy | Runtime |
|---|---:|---:|
| Backtracking | 100% puzzles | ~594 ms avg |
| CNN | 32.53% cells / 0% puzzles | 0.080 ms avg |

Backtracking reliably solves the puzzles but can have highly variable
runtime (1.6 ms–23.7 s). The CNN is much faster but cannot produce
complete valid solutions, motivating the hybrid approach.

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
```bash
# Prepare dataset
python -m scripts.prepare_dataset

# Evaluate backtracking
python -m scripts.evaluate

# Train neural baseline
python -m src.neural.train

# Evaluate neural baseline
python -m scripts.evaluate_neural
```
## To-do 
- Implement neural-guided symbolic constraint solving
- Build the hybrid neuro-symbolic solver
- Compare neural, backtracking, and hybrid puzzle accuracy
- Evaluate generalisation across puzzle difficulties
