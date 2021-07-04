import tictactoe as ttt
import numpy as np


# Testing next move
a = ttt.initial_state()
a = ttt.result(a, (1, 1))
a = ttt.result(a, (0, 0))
a = ttt.result(a, (2, 0))
a = ttt.result(a, (0, 2))
