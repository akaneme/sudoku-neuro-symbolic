def find_empty(board):
    """return (row, col) of an empty cell, or None if full."""
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                return row, col

    return None


def is_valid(board, row, col, num):
    """check whether num can be placed at board[row][col]."""

    # check row
    for c in range(9):
        if board[row][c] == num:
            return False

    # check column
    for r in range(9):
        if board[r][col] == num:
            return False

    # check 3x3 box
    box_row = (row // 3) * 3
    box_col = (col // 3) * 3

    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if board[r][c] == num:
                return False

    return True


def solve(board):
    """solve Sudoku in-place using recursive backtracking."""

    empty = find_empty(board)

    # no empty cells -> solved
    if empty is None:
        return True

    row, col = empty

    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row][col] = num

            if solve(board):
                return True

            # wrong choice -> undo
            board[row][col] = 0

    return False


def string_to_board(puzzle):
    """convert an 81-character puzzle string to a 9x9 board."""

    return [
        [
            0 if char == "." else int(char)
            for char in puzzle[row * 9:(row + 1) * 9]
        ]
        for row in range(9)
    ]


def board_to_string(board):
    """convert a 9x9 board back to an 81-character string."""

    return "".join(
        str(cell)
        for row in board
        for cell in row
    )


def solve_puzzle(puzzle):
    """solve an 81-character Sudoku puzzle string."""

    board = string_to_board(puzzle)

    if solve(board):
        return board_to_string(board)

    return None