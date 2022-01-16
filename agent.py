#agent 
import minesweeper as ms
import numpy as np
from collections import deque
import torch
import random
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001

GAME_SIZE = 9
MINE_COUNT = 10

class Agent:
    def __init__(self):
        self.gamma = 0
        self.epsilon = 0
        self.number_of_games = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(GAME_SIZE*GAME_SIZE, 256, GAME_SIZE*GAME_SIZE)
        self.trainer = QTrainer(self.model, LEARNING_RATE, self.gamma)

    def getAction(self, state, game_size, game_instance: ms.Minesweeper):
        self.epsilon = 80 - self.number_of_games
        move = (0,0)
        if random.randint(0, 200) < self.epsilon:
            while True:
                i = random.randint(0, game_size-1)
                j = random.randint(0, game_size-1)
                if not game_instance.field[i,j]:
                    move = (i,j)
                    break
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            _move = torch.argmax(prediction).item()
            move = np.unravel_index(move, game_instance.field.shape)

        return move

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
        self.trainer.train_step(states,- new_states, moves, rewards, dones)


    def getState(self, game_instance: ms.Minesweeper):
        out = np.full(game_instance.field_assignment.shape, -1)

        for i in range(0, game_instance.game_size):
            for j in range(0, game_instance.game_size):
                if game_instance.field[i,j]:
                    out[i,j] = game_instance.field_assignment[i,j]

        return out


def train():
    high_score = 0
    total_score = 0
    plot_scores = []
    plot_mean_scores = []
    agent = Agent()
    game = ms.Minesweeper(GAME_SIZE, MINE_COUNT)

    while True:
        state = agent.getState(game)
        move = agent.getAction(state, GAME_SIZE, game) #the next move aka field to click on
        old_score = game.unfolded
        isLost = game.unfold(move[0], move[1])
        score = game.unfolded
        reward = (score - old_score) #mehr reward bei mehr feldern, die auf einmal aufgedeckt wurden?

        state_new = agent.getState(game)

        agent.train_short_memory(state, state_new, move, reward, isLost)

        agent.remember(state, state_new, move, reward, isLost)

        if isLost:
            game = ms.Minesweeper(GAME_SIZE, MINE_COUNT)
            agent.number_of_games += 1
            agent.train_long_memory()
        
            if high_score < score:
                high_score = score
                agent.model.save()

            print("Game", agent.number_of_games, "score", score, "highscore", high_score)

            

if __name__ == "__main__":
    train()