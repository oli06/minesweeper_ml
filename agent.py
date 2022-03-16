
import numpy as np
import random
from model import QTrainer
from bruteforce import BruteforceMinesweeperObject
import minesweeper as ms




class Agent:
    def __init__(self, game_size, learning_rate, epsilon, gamma, max_memory, model_path=""):
        self.game_size = game_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.number_of_games = 0
        self.trainer = QTrainer(learning_rate, self.gamma, self.epsilon, max_memory, model_path=model_path)


    def getAction(self, state, game_size, game: ms.Minesweeper, bruteforceInstance: BruteforceMinesweeperObject):
        '''
        Based on the qtrainer-epsilon value, this function returns the next move, 
        which is either choosen randomly, using bruteforce or predicted by the model.

                Parameters:
                        state: the current state of the game
                        game_size (int): game_size
                        game (Minesweeper): the Minesweeper game, which is currently played
                        bruteforceInstance (BruteforceMinesweeperObject): a bruteforce instance which is used to draw the next move

                Returns:
                        a move (i,j) followed by a boolean, which is True, if it is a real random move, False otherwise
        '''
        if random.random() < self.trainer.epsilon:
            bruteforce_move = bruteforceInstance.bruteforce_prediction(game.field, game.field_assignment)
            if bruteforce_move:
                return bruteforce_move, False

            while True:
                #regenerate field if already unfolded
                i = random.randint(0, game_size-1)
                j = random.randint(0, game_size-1)
                if not game.field[i,j]: 
                    move = (i,j)
                    return move, True

        prediction = self.trainer.model.predict(np.reshape(state, (1, game_size, game_size, 2))) #since our model always uses a input of (None, game_size, game_size, 2) where None is a variable batch size (in this prediction case it must be 1), we need to add another dimension, such that (1,9,9,2)
            #therefore prediction is an array (of batch_size), since our batch_size = 1, we need the first prediction
        prediction[np.reshape(game.field, (1,self.game_size*self.game_size)) == True] = np.min(prediction)
        move = np.unravel_index(np.argmax(prediction), game.field.shape)
        return move, False

    def _getActions(self, game_instance: ms.Minesweeper):
        '''
        Returns all possible actions for a given game

                Parameters:
                        game (Minesweeper): a Minesweeper instance

                Returns:
                        array of shape game.field indicating all available actions
        '''
        return np.invert(game_instance.field)

    def train_short_memory(self, is_game_lost):
        self.trainer.train_step(is_game_lost)

    def remember(self, state, state_new, move, reward, done):
        '''
        Stores the progress in the memory

                Parameters:
                        state: the state before the move
                        state_new: the new state after the move
                        move (i,j): the move
                        reward (int): the taken reward based on the move
                        done (boolean): True, if the game is lost or won, False otherwise

                Returns:
                        void
        '''
        self.trainer.memory.append((state, state_new, move, reward, done))

    def getState(self, game_instance: ms.Minesweeper):
        '''
        Returns the state of the game as a easy to process array

                Parameters:
                        game (Minesweeper): a Minesweeper instance
                        
                Returns:
                        the current state of the game
        '''
        result = np.zeros((game_instance.game_size, game_instance.game_size, 2))
        filter = ~np.logical_or(game_instance.field == False, game_instance.field_assignment == 0) #Not U or E
        result[filter, 0] = game_instance.field_assignment[filter]
        result[game_instance.field == False, 1] = 1
        return result

    def calculate_reward(self, game:ms.Minesweeper, done, field_already_unfolded, old_score, has_neighbors, already_unfolded_multiplier):
        '''
        Returns the reward for a given move

                Parameters:
                        game (Minesweeper): the Minesweeper instance
                        done (boolean): Indicator if the game is lost or won, False otherwise
                        field_already_unfolded (boolean): Indicator if the field is already unfolded
                        old_score (int): score of the game before the move was taken
                        has_neigbors (boolean): Indicator if the unfolded field has unfolded neighbours
                        already_unfolded_multiplier (int): if a move is taken which was taken before, the rewards get punished even more

                Returns:
                        the reward of the move based on the given parameters / conditions
        '''
        reward = .3
        if field_already_unfolded:
            reward = -0.3 * already_unfolded_multiplier
        elif done:
            reward = 1 if game.is_game_won() else -1
        elif old_score > 0 and not has_neighbors:
            reward = -.3

        return reward