import math
from random import randint
from minesweeper import Minesweeper


def initial_bruteforce_unfolding(game: Minesweeper, number_of_unfolded_fields: int):
        i = randint(0,8)
        j = randint(0,8)
        game.unfold(i,j)
        left = max(0, j-1)
        right = min(j + 1, 9-1)
        top = max(0, i-1)
        bottom = min(i+1, 9-1)
        for o in range(left, right+1):
            for p in range(top, bottom+1):
                if game.unfolded > number_of_unfolded_fields:
                    return game
                if not game.field[p, o]:
                    if game.unfold(p, o):
                        #verloren... neues game generieren und von vorne beginnen...
                        return initial_bruteforce_unfolding(Minesweeper.Minesweeper(game.game_size, game.mines))

        return game

def pretty_print(self, field, field_assignment):
    for i in range(0,self.game_size_x):
        print_str = " "
        for j in range(0,self.game_size_y):
            if self.mines_for_sure[i,j]:
                if not field_assignment[i,j] == math.inf:
                    print("STH WENT TOtally wrong")
                else:
                    print_str += " B "
            else: 
                print_str += " U " if not field[i,j] else " " + str(int(field_assignment[i,j])) + " "
        print(print_str)