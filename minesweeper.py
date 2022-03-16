#minesweeper logic

import numpy as np
import random as rand
import math

class Minesweeper:
    def __init__(self, game_size, mines):
        self.game_size = game_size
        self.mines = mines
        self.unfolded = 0
        self.__generate_game()


    def __generate_game(self):
        '''
        Instantiates a new minesweeper game with size self.game_size and #mines of self.mines.
        Initializing self.fields and self.field_assignment.
        '''
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

        #print(field_assignment)

    def unfold(self, i, j):
        '''
        For given coordinates (i,j), the field is unfolded. 
        If the field value == 0, all neighbours are unfolded as well.
        If the field value == math.inf and it is the first move, the game is reinitalized.
        If it is not the first move, the game is lost.

                Parameters:
                        i (int): y-coordinate of the field to unfold
                        j (int): x-coordinate of the field to unfold

                Returns:
                        True, if the game is won OR lost, False otherwise
        '''
        assert i < self.game_size and j < self.game_size

        if self.field[i][j]:
            return False #field already unfolded

        if self.field_assignment[i][j] == math.inf:
            if not self.field.any(): #first click
                #if the first click lands on a bomb, we simply regenerate the game as long as we dont generate a mine on i,j
                self.__generate_game()
                while self.unfold(i,j):
                    self.__generate_game()

                return False
            else:
                return True #you lost the game / selected a mine


        self.field[i][j] = 1
        if self.field_assignment[i][j] == 0:
            #unfold all zero-fields connected to this one
            self.unfold_neighbors(i,j)

        self.unfolded += 1
        #print(f"unfolded {i},{j} ist {self.field_assignment[i,j]}")

        if np.sum(self.field == False) == self.mines:
            #game is won
            return True
        
        return False

    def is_game_won(self):
        '''
        Returns True, if the game is won, False otherwise
        '''
        return np.sum(self.field == False) == self.mines

    def unfold_neighbors(self, i, j):
        '''
        For given coordinates (i,j), all its neighbours are unfolded.

                Parameters:
                        i (int): y-coordinate of the field
                        j (int): x-coordinate of the field

                Returns:
                        void
        '''
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
