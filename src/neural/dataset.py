import torch
from torch.utils.data import Dataset


class SudokuDataset(Dataset):

    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        puzzle = self.string_to_tensor(row["puzzle"])
        solution = self.string_to_tensor(row["solution"])

        return puzzle, solution

    @staticmethod
    def string_to_tensor(sudoku_string):
        sudoku_string = str(sudoku_string)
        values = [
            0 if char == "." else int(char)
            for char in sudoku_string
        ]

        return torch.tensor(
            values,
            dtype=torch.long,
        ).reshape(9, 9)