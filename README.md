# Neuro-Symbolic Sudoku Solver

Can a neural network solve puzzles it has never seen, 
if it knows the rules?

This project compares three approaches to solving Sudoku:
- Classical backtracking (symbolic baseline)
- Pure neural solver (CNN/MLP)
- Hybrid: neural solver with symbolic constraint injection

The goal is to study whether explicitly encoding 
the rules of Sudoku into a neural solver improves 
generalization on harder, unseen puzzle difficulties 
compared to a purely learned approach.

## Status
Work in progress. Backtracking baseline complete.
Neural solver in development.

## Stack
Python, PyTorch, NumPy

## Structure (planned)
backtracking/ — symbolic baseline
neural/ — CNN/MLP solver
hybrid/ — constraint-injected model
data/ — puzzle datasets by difficulty
