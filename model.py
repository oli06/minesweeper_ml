from collections import deque
from pickle import load
import random
import torch.nn as nn
import numpy as np
from tensorflow import keras
from dqn import create_dqn

BATCH_SIZE = 64
MIN_TRAIN_SIZE = 1_000

LEARNING_RATE = 0.01
LEARNING_RATE_MIN = 0.001
UPDATE_NEXT_STATE_MODEL_AT = 5
GAME_SIZE = 9
EPSILON_DECREASE = 0.99975
LEARNING_RATE_DECREASE = 0.99975
EPSILON_MIN = 0.01
CONV_UNITS = 64 # number of neurons in each conv layer
DENSE_UNITS = 512 # number of neurons in fully connected dense layer

class QTrainer:
    def __init__(self, lr, gamma, epsilon, MAX_MEMORY_SIZE, load_model=False):
        self.gamma = gamma
        self.learning_rate = lr
        self.next_state_model_counter = 0
        self.criterion = nn.MSELoss()
        self.epsilon = epsilon

        if load_model:
            self.model = keras.models.load_model('models/model_4000.h5')
            self.next_state_model = self.model
            with open("models/model_4000.pkl", "rb") as f:
                self.memory = load(f)
        else:
            self.memory = deque(maxlen=MAX_MEMORY_SIZE)
            self.model = create_dqn(LEARNING_RATE, (GAME_SIZE,GAME_SIZE,2), GAME_SIZE*GAME_SIZE, CONV_UNITS, DENSE_UNITS)
            self.next_state_model = create_dqn(LEARNING_RATE, (GAME_SIZE,GAME_SIZE,2), GAME_SIZE*GAME_SIZE, CONV_UNITS, DENSE_UNITS)


    def train_step(self, update_next_state_model):
        if len(self.memory) < MIN_TRAIN_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states = np.array([single_game[0] for single_game in batch])
        q = self.model.predict(states)

        new_states = np.array([single_game[1] for single_game in batch])
        new_q_list = self.next_state_model.predict(new_states)

        for i, (_, _, move, reward, done) in enumerate(batch):
            new_q = reward
            if not done:
                new_q += self.gamma * np.max(new_q_list[i])
            
            move_index = GAME_SIZE * move[0] + move[1]
            q[i][move_index] = new_q

        self.model.fit(np.array(states), np.array(q), batch_size=BATCH_SIZE, shuffle=False, verbose=0)

        if update_next_state_model:
            self.next_state_model_counter += 1

        if self.next_state_model_counter > UPDATE_NEXT_STATE_MODEL_AT:
            self.next_state_model.set_weights(self.model.get_weights())
            self.next_state_model_counter = 0

        #update learning_rate, epsilon
        self.epsilon = max(EPSILON_MIN, self.epsilon*EPSILON_DECREASE)
        self.learning_rate = max(LEARNING_RATE_MIN, self.learning_rate*LEARNING_RATE_DECREASE)