"""
Tic Tac Toe Player
"""

import math
import numpy as np

X = 1
O = 0
EMPTY = -1


def initial_state():
    """
    Returns starting state of the board.
    """
    return np.full((3, 3), -1, dtype=np.int8)


def player(board: np.ndarray):
    """
    Returns player who has the next turn on a board.
    """
    return np.count_nonzero(board == -1) % 2


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    return list(zip(*np.where(board == -1)))


def result(board: np.ndarray, action: tuple[int, int]):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    new_board = board.copy()
    new_board[action[0], action[1]] = player(board)
    return new_board


def winner(board: np.ndarray):
    """
    Returns the winner of the game, if there is one.
    """
    col = np.where((board.min(0) == board.max(0)) & (board.min(0) != -1))[0]
    if len(col) > 0:
        return board[0, col[0]]

    row = np.where((board.min(1) == board.max(1)) & (board.min(1) != -1))[0]
    if len(row) > 0:
        return board[row[0], 0]

    left_diagonal = board.diagonal()
    if (left_diagonal.min() == left_diagonal.max()) and (left_diagonal[0] != -1):
        return left_diagonal[0]

    right_diagonal = np.fliplr(board).diagonal()
    if (right_diagonal.min() == right_diagonal.max()) and (right_diagonal[0] != -1):
        return right_diagonal[0]


def terminal(board: np.ndarray):
    """
    Returns True if game is over, False otherwise.
    """
    return winner(board) != None or (np.count_nonzero(board == -1) == 0)


def utility(board: np.ndarray):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    result = winner(board)
    if result == 1:
        return 1
    elif result == 0:
        return -1
    else:
        return 0


def minimax(board: np.ndarray):
    """
    Returns the optimal action for the current player on the board.
    """
    best_action = [-1, -1]
    if player(board):
        v = -np.inf
        for action in actions(board):
            res = min_value(result(board, action))
            v, best_action = (res, action) if v < res else (v, best_action)
    else:
        v = np.inf
        for action in actions(board):
            res = max_value(result(board, action))
            v, best_action = (res, action) if v > res else (v, best_action)

    return best_action


def max_value(board):
    if terminal(board):
        return utility(board)
    v = -np.inf
    for action in actions(board):
        v = max(v, min_value(result(board, action)))
    return v


def min_value(board):
    if terminal(board):
        return utility(board)
    v = np.inf
    for action in actions(board):
        v = min(v, max_value(result(board, action)))
    return v
