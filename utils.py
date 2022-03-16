import math
from random import randint
from minesweeper import Minesweeper


def initial_bruteforce_unfolding(game: Minesweeper, number_of_unfolded_fields: int):
    '''
    Unfolds number_of_unfolded_fields without selecting mines.

            Parameters:
                    game (Minesweeper): The game which is played
                    number_of_unfolded_fields (int): The number of fields which are unfolded after the call

            Returns:
                    void
    '''
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
    '''
    Prints a Minesweeper Game as a human-readable board

            Parameters:
                    field (Array): Array of unfolded fields
                    field_assignment: (Array): Array of field values

            Returns:
                    void
    '''
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

def has_neighbour(move, game):
    '''
    Returns True, if any direct neighbour of move is already unfolded

            Parameters:
                    move (tuple): A minesweeper coordination tuple containing the field to look for neighbour
                    game (Minesweeper): A minesweeper game

            Returns:
                    True if any neighbour is already unfolded, False otherwise
    '''
    #calculating neigbor indices
    left = max(0, move[0]-1)
    right = max(0, move[0]+2)
    bottom = max(0, move[1]-1)
    top = max(0, move[1]+2)

    if game.field[left:right,bottom:top].any():
        return True
    return False