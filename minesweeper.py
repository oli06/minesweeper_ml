#minesweeper logic

import numpy as np
import random as rand
import math

class Game:
    def __init__(self, game_size, mines):
        self.game_size = game_size
        self.mines = mines
        self.solved = self.game_size ** 2 - self.mines #TODO: rename, number of required correct "guesses" to solve and win the game
        self.unfolded = 0
        self.__generate_game()


    def __generate_game(self):
        #init game data

        #generate mines locations and update neighbors
        field_assignment = np.zeros((self.game_size, self.game_size))
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
        self.field = np.zeros((self.game_size, self.game_size))

        print(field_assignment)

    def unfold(self, i, j, checkAvailableMoves=True):
        assert i < self.game_size and j < self.game_size

        if self.field[i][j]:
            return True #field already unfolded

        if checkAvailableMoves and not self.move_available():
            pass
            return 


        if self.field_assignment[i][j] == math.inf:
            return False #you lost the game / selected a mine

        self.field[i][j] = 1

        if self.field_assignment[i][j] == 0:
            #unfold all zero-fields connected to this one
            self.unfold_neighbors(i,j)

        self.unfolded += 1
        return True

    def is_game_won(self):
        return self.solved == self.unfolded


    def unfold_neighbors(self, i, j):
        #unfold all neigbors that are zeros and habe a number

        left = max(0, i-1)
        right = max(0, i+2)
        bottom = max(0, j-1)
        top = max(0, j+2)
        
        #unfold all connected neighbors of a 0-field
        for o in range(left, right):
            for p in range(bottom, top):
                if (o != i or p != j) and not self.field[o][p]:
                    self.unfold(o, p, False) #ignore check for available random fields

    #TODO
    def move_available(self):
        #return true, if there is at least one possible move (based on self.filed/ self.assignment) available by calculation
        #return false, if a random guess is required

        return True

g = Game(9, 10)
x = int(input())
y = int(input())

g.unfold(x,y)
print(g.field)
print(g.field_assignment)