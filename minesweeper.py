#minesweeper logic

import numpy as np
import random as rand
import math

class Minesweeper:
    def __init__(self, game_size, mines):
        self.game_size = game_size
        self.mines = mines
        self.solved = self.game_size ** 2 - self.mines #TODO: rename, number of required correct "guesses" to solve and win the game
        self.unfolded = 0
        self.__generate_game()


    def __generate_game(self):
        #init game data

        #generate mines locations and update neighbors
        field_assignment = np.zeros((self.game_size, self.game_size), dtype=np.float)
        mines_count = 0
        while mines_count < self.mines:
            next_i = rand.randint(0, self.game_size-1)
            next_j = rand.randint(0, self.game_size-1)
            
            if field_assignment[next_i][next_j] == 0:
                field_assignment[next_i][next_j] = math.inf
                mines_count += 1

                #calculating neigbor indices
                left = max(0, next_i-1)
                right = max(0, next_i+2)
                bottom = max(0, next_j-1)
                top = max(0, next_j+2)

                field_assignment[left:right,bottom:top] += 1 #update all neighbors (including the field itself), since mines are stored as math.inf there is no impact on such fields

        self.field_assignment = field_assignment
        self.field = np.zeros((self.game_size, self.game_size), dtype=np.bool_)

        print(field_assignment)

    def unfold(self, i, j):
        assert i < self.game_size and j < self.game_size

        if self.field[i][j]:
            return True #field already unfolded

        if self.field_assignment[i][j] == math.inf:
            if not self.field.any(): #first click
                #if the first click lands on a bomb, we simply regenerate the game as long as we dont generate a mine on i,j
                self.__generate_game()
                while not self.unfold(i,j):
                    self.__generate_game()

                return True
            else:
                return False #you lost the game / selected a mine

        self.field[i][j] = 1

        if self.field_assignment[i][j] == 0:
            #unfold all zero-fields connected to this one
            self.unfold_neighbors(i,j)

        self.unfolded += 1
        print(f"unfolded {i},{j} ist {self.field_assignment[i,j]}")
        return True

    def is_game_won(self):
        return self.solved == self.unfolded


    def unfold_neighbors(self, i, j):
        #unfold all neigbors that are zeros and habe a number

        left = max(0, j-1)
        right = min(j + 1, self.game_size-1)
        top = max(0, i-1)
        bottom = min(i+1, self.game_size-1)
        
        #unfold all connected neighbors of a 0-field
        for o in range(left, right+1):
            for p in range(top, bottom+1):
                if (o != j or p != i) and not self.field[p, o]:
                    self.unfold(p, o) #ignore check for available random fields
