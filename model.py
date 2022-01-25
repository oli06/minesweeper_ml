from collections import deque
from pickle import load
import random
from turtle import update
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

from dqn import create_dqn

BATCH_SIZE = 64
MIN_TRAIN_SIZE = 1_000

LEARNING_RATE = 0.01
LEARNING_RATE_MIN = 0.001
UPDATE_NEXT_STATE_MODEL_AT = 5
GAME_SIZE = 9
MINE_COUNT = 10
EPSILON_DECREASE = 0.99975
LEARNING_RATE_DECREASE = 0.99975
EPSILON_MIN = 0.01
CONV_UNITS = 64 # number of neurons in each conv layer
DENSE_UNITS = 512 # number of neurons in fully connected dense layer


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        #self.linear1 = nn.Linear(input_size, hidden_size)
        #self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        #x = F.relu(self.linear1(x))
        #x = self.linear2(x)
        x = F.relu(self.conv1(x))
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)



class QTrainer:
    def __init__(self, lr, gamma, epsilon, MAX_MEMORY_SIZE, load_model=False):
        self.gamma = gamma
        self.learning_rate = lr
        if load_model:
            self.model = keras.models.load_model('models/model_4000.h5')
            self.next_state_model = self.model
            with open("models/model_4000.pkl", "rb") as f:
                self.memory = load(f)
        else:
            self.memory = deque(maxlen=MAX_MEMORY_SIZE)
            self.model = create_dqn(LEARNING_RATE, (GAME_SIZE,GAME_SIZE,2), GAME_SIZE*GAME_SIZE, CONV_UNITS, DENSE_UNITS)
            self.next_state_model = create_dqn(LEARNING_RATE, (GAME_SIZE,GAME_SIZE,2), GAME_SIZE*GAME_SIZE, CONV_UNITS, DENSE_UNITS)
        self.next_state_model_counter = 0
        #self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.epsilon = epsilon

    def train_step(self, update_next_state_model):
        if len(self.memory) < MIN_TRAIN_SIZE:
            return
        """state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        move = torch.tensor(move, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 2:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            move = torch.unsqueeze(move, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
        """

        #state = tf.convert_to_tensor(state, np.float32)
        #next_state = tf.convert_to_tensor(next_state, np.float32)
        #move = tf.convert_to_tensor(move, np.int64)
        #reward = tf.convert_to_tensor(reward, np.float32)
        # (n, x)

        batch = random.sample(self.memory, BATCH_SIZE)
        states = np.array([single_game[0] for single_game in batch])
        q = self.model.predict(states)

        new_states = np.array([single_game[1] for single_game in batch])
        new_q_list = self.next_state_model.predict(new_states)

        y = []

        for i, (state, new_state, move, reward, done) in enumerate(batch):
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
"""
        if not longterm_training and len(state.shape) == 3:
            # (1, x), if it is a single sample (i.e first iteration, we need to append an iterable dimension) 
            state = tf.expand_dims(state, 0)
            next_state = tf.expand_dims(next_state, 0)
            move = tf.expand_dims(move, 0)
            reward = tf.expand_dims(reward, 0)
            done = (done, )
        elif longterm_training:
            state = tf.convert_to_tensor(state, np.float32)
            next_state = tf.convert_to_tensor(next_state, np.float32)
            move = tf.convert_to_tensor(move, np.int64)
            reward = tf.convert_to_tensor(reward, np.float32)

        # 1: predicted Q values with current state
        pred = self.model.predict(state)
        X,y = [], []
        target = pred.copy()
        #Q_new = reward + (self.gamma * tf.reduce_max(self.model.predict(next_state), axis=1) if not done else 0)
        Q_new = reward
        idx = move[:,0] * 9 + move[:,1]
        target[:,idx] = Q_new

        #for idx in range(len(done)):
            #Q_new = reward[idx]
            #if not done[idx]:
                #Q_new = reward[idx] + self.gamma * tf.reduce_max(self.model.predict(tf.expand_dims(next_state[idx], 0)))

            #index = move[0][0] * 9 + move[0][1]

            ##target[idx][0][torch.argmax(move[idx]).item()] = Q_new
            ##target[idx][0][index.item()] = Q_new
            #target[idx][index] = Q_new
         
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        

        self.model.fit(state, target, batch_size=64,
                       shuffle=False, verbose=0)

        
        #self.optimizer.zero_grad()
        #loss = self.criterion(target, pred)
        #loss.backward()

        #self.optimizer.step()
        

"""