import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import tensorflow as tf

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
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        #self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, next_state, move, reward, done):
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

        state = tf.convert_to_tensor(state, np.float32)
        next_state = tf.convert_to_tensor(next_state, np.float32)
        move = tf.convert_to_tensor(move, np.int64)
        reward = tf.convert_to_tensor(reward, np.float32)
        # (n, x)

        if len(state.shape) == 3:
            # (1, x)
            state = tf.expand_dims(state, 0)
            next_state = tf.expand_dims(next_state, 0)
            move = tf.expand_dims(move, 0)
            reward = tf.expand_dims(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model.predict(state)
        X,y = [], []
        target = pred.copy()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * tf.reduce_max(self.model.predict(tf.expand_dims(next_state[idx], 0)))

            index = move[0][0] * 9 + move[0][1]

            #target[idx][0][torch.argmax(move[idx]).item()] = Q_new
            #target[idx][0][index.item()] = Q_new
            target[idx][index] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        

        self.model.fit(np.array(state), np.array(target), batch_size=64,
                       shuffle=False, verbose=0)

        """
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
        """
