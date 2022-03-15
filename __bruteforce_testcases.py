import numpy as np
import math
from bruteforce import BruteforceMinesweeperObject
from gui import MinesweeperUI

from utils import initial_bruteforce_unfolding, pretty_print

"""
TESTCASE 1 6x6 Game
"""
arr = np.array([[ 0., 1,  1,  2, 1,  1],
       [ 1,  2,  math.inf,  2,  math.inf,  2.],
       [ math.inf,  3,  3,  4,  3,  math.inf],
       [1,  2,  math.inf,  math.inf,  2,  1.],
       [1 ,  2,  2,  2, 2,  1.],
       [math.inf,  1.,  0,  0.,  1.,  math.inf]
       ])
f_arr = np.array([
     [False, False, False, False, False, False],
      [False, False, False, True, False, False],
       [False, False, False, True, False, False],
        [False, True, False, False, True, False],
         [False, True, True, True, True, False],
          [False, True, True, True, True, False]
      ])
pretty_print(f_arr, arr, 6)
bf_instance = BruteforceMinesweeperObject(6,6)
safe_guess = bf_instance.bruteforce_prediction(f_arr, arr)
while not safe_guess == None:
    print(safe_guess)
    print(bf_instance.safe_for_sure)
    f_arr[safe_guess[0], safe_guess[1]] = True
    pretty_print(f_arr, arr,6)
    safe_guess = bf_instance.bruteforce_prediction(f_arr, arr)





"""
TESTCASE 1 - 1000
"""

bruteforce_won_games = []
not_solved = []
lost_games = []
for i in range(1000):
    safe_for_sure = []
    mines_for_sure = np.zeros((9,9))
    danger_zone = []
    game = initial_bruteforce_unfolding(MinesweeperUI.Minesweeper(9, 10))
    bf_instance = BruteforceMinesweeperObject(9,9)
    move = bf_instance.bruteforce_prediction(game.field, game.field_assignment)

    done = False
    while move:
        if game.unfold(move[0], move[1]):
            done = True
            if game.is_game_won():
                for i in range(0,9):
                    for j in range(0,9):
                        if game.field[i,j]:
                            if game.field_assignment[i,j] == math.inf:
                                print("fehler")
                        if not game.field[i,j]:
                            if game.field_assignment[i,j] != math.inf:
                                print("fehler too")
                bruteforce_won_games.append(game)
            else:
                lost_games.append((game, move, mines_for_sure, safe_for_sure, danger_zone))
            break
        move = bf_instance.bruteforce_prediction(game.field, game.field_assignment)
    if not done:
        not_solved.append(game)

print("done")