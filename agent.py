#agent 
from cmath import inf
from dqn import create_dqn
import minesweeper as ms
import numpy as np
from collections import deque
import torch
import random
from model import Linear_QNet, QTrainer
import tensorflow as tf

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001

GAME_SIZE = 9
MINE_COUNT = 10

CONV_UNITS = 64 # number of neurons in each conv layer
DENSE_UNITS = 512 # number of neurons in fully connected dense layer

class Agent:
    def __init__(self):
        self.gamma = 0.1
        self.epsilon = 1
        self.number_of_games = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        #self.model = Linear_QNet(GAME_SIZE*GAME_SIZE, 256, GAME_SIZE*GAME_SIZE)
        self.model = create_dqn(LEARNING_RATE, (GAME_SIZE,GAME_SIZE,2), GAME_SIZE*GAME_SIZE, CONV_UNITS, DENSE_UNITS)
        self.trainer = QTrainer(self.model, LEARNING_RATE, self.gamma)

    def __update_epsilon(self):
        self.epsilon = self.epsilon - 0.03 #after 3000 games we reached our epsilon-minimum
        if self.epsilon < 0.1: #minimum of 0.1
            self.epsilon = 0.1
        

    def getAction(self, state, game_size, game_instance: ms.Minesweeper):
        if self.number_of_games % 100 == 0:
            self.__update_epsilon()
        self.epsilon = 80 - self.number_of_games
    
        if random.randint(0,200) < self.epsilon:
            while True:
                i = random.randint(0, game_size-1)
                j = random.randint(0, game_size-1)
                if not game_instance.field[i,j]:
                    move = (i,j)
                    break
        
            return move, True

        state0 = tf.convert_to_tensor(state, dtype=np.float32)
        prediction = self.model.predict(tf.expand_dims(state0, 0))
        argmax_move = tf.math.argmax(prediction[0])
        move = np.unravel_index(argmax_move, game_instance.field.shape)
        return move, False

    def _getActions(self, game_instance: ms.Minesweeper):
        return np.invert(game_instance.field)

    def train_short_memory(self, state, state_new, move, reward, done):
        self.trainer.train_step(state, state_new, move, reward, done)

    def remember(self, state, state_new, move, reward, done):
        self.memory.append((state, state_new, move, reward, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, new_states, moves, rewards, dones = zip(*mini_sample)
        self.trainer.train_step(states, new_states, moves, rewards, dones)


    def getState(self, game_instance: ms.Minesweeper):
        result = np.zeros((game_instance.game_size, game_instance.game_size, 2))
        filter = ~np.logical_or(game_instance.field == False, game_instance.field_assignment == 0) #Not U or E
        result[filter, 0] = game_instance.field_assignment[filter] / 4
        result[game_instance.field == False, 1] = 1
        return result
        #one-hot representation NxNx10
        #eight fields, one-hot encoded for the number (1-8)
        #one field indicating a zero-value field
        #one field is 1 if folded otherwise 0

        """n_values = 11
        out = np.eye(n_values)[game_instance.field_assignment[game_instance.field_assignment == math.inf] = 0]

        for i in range(0, game_instance.game_size):
            for j in range(0, game_instance.game_size):
                if not game_instance.field[i,j]:
                    out[i,j][n_values-1] = 1
                if game_instance.field_assignment[i,j] == 0:
                    out[i,j][n_values-2] = 1
        """
        """out = np.full(game_instance.field_assignment.shape, -1)

        for i in range(0, game_instance.game_size):
            for j in range(0, game_instance.game_size):
                if game_instance.field[i,j]:
                    #out[i,j] = game_instance.field_assignment[i,j]
                    out[i,j] = 1 #anstatt die tatsaechlichen werte aus dem spiel zu verwenden, setzen wir aufgedeckt=1

        return out.reshape((1,81))
        """
        return out
                
                


def has_neighbour(move, game):
    #calculating neigbor indices
    left = max(0, move[0]-1)
    right = max(0, move[0]+2)
    bottom = max(0, move[1]-1)
    top = max(0, move[1]+2)

    if game.field[left:right,bottom:top].any():
        return True
    return False

def train():
    high_score = 0
    total_score = 0
    plot_scores = []
    plot_mean_scores = []
    agent = Agent()
    game = ms.Minesweeper(GAME_SIZE, MINE_COUNT)
    random_counter = 0
    field_already_unfolded_multiplier = 1

    while True:
        state = agent.getState(game)
        move, random_move = agent.getAction(state, GAME_SIZE, game) #the next move aka field to click on
        random_counter += random_move

        old_score = game.unfolded
        field_already_unfolded = game.field[move[0], move[1]]
        has_unfolded_neighbours = has_neighbour(move, game)

        isLost = not game.unfold(move[0], move[1])
        score = game.unfolded
        is_game_won = game.is_game_won()

        reward = 0.7
        if field_already_unfolded:
            #reward = -2 * field_already_unfolded_multiplier 
            #field_already_unfolded_multiplier += 1
            reward = -0.3
        elif not has_unfolded_neighbours and old_score == 0:
            reward = -0.3
        elif isLost:
            reward = -1
        elif is_game_won:
            reward = 5
        
        state_new = agent.getState(game)

        agent.train_short_memory(state, state_new, move, reward, isLost)

        agent.remember(state, state_new, move, reward, isLost)

        if game.is_game_won() or isLost:
            game = ms.Minesweeper(GAME_SIZE, MINE_COUNT)
            agent.number_of_games += 1
            agent.train_long_memory()
        
            if high_score < score:
                high_score = score
                #agent.model.save()


            total_score += score
            if game.is_game_won():


                print(f"+++++++++ WIR HABEN GEWONNEN🔫🔫🔫 +++++++")
                print(f"+++++++++ WIR HABEN GEWONNEN🔫🔫🔫 +++++++")

                print(f"+++++++++ WIR HABEN GEWONNEN🔫🔫🔫 +++++++")

                print(f"+++++++++ WIR HABEN GEWONNEN🔫🔫🔫 +++++++")

                print(f"+++++++++ WIR HABEN GEWONNEN🔫🔫🔫 +++++++")

                print(f"+++++++++ WIR HABEN GEWONNEN🔫🔫🔫 +++++++")
                print()

            print("Game", agent.number_of_games, "score", score, "highscore", high_score, "average score", total_score / agent.number_of_games, "random move", random_counter)
            random_counter = 0
            field_already_unfolded_multiplier = 1
            

if __name__ == "__main__":
    train()