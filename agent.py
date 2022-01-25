from dqn import create_dqn
#agent 
import pickle

from tqdm import tqdm
import minesweeper as ms
import numpy as np
import random
from model import QTrainer
import tensorflow as tf

MAX_MEMORY = 25_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001
AGG_STATS_EVERY = 100 # calculate stats every 100 games for tensorboard
SAVE_MODEL_EVERY = 1000 # save model and replay every 10,000 episodes

GAME_SIZE = 9
MINE_COUNT = 10

CONV_UNITS = 64 # number of neurons in each conv layer
DENSE_UNITS = 512 # number of neurons in fully connected dense layer

class Agent:
    def __init__(self, load_model=False):
        self.gamma = 0.1
        self.epsilon = 1
        self.number_of_games = 0
        self.trainer = QTrainer(LEARNING_RATE, self.gamma, self.epsilon, MAX_MEMORY, load_model)


    def getAction(self, state, game_size):
        if random.random() < self.epsilon:
            while True:
                i = random.randint(0, game_size-1)
                j = random.randint(0, game_size-1)
                if state[i,j, 1]: #regenerate field if already unfolded
                    move = (i,j)
                    break
        
            return move, True

        #state0 = tf.convert_to_tensor(state, dtype=np.float32)
        #prediction = self.model.predict(state0)
        prediction = self.trainer.model.predict(np.reshape(state, (1, game_size, game_size, 2))) #since our model always uses a input of (None, game_size, game_size, 2) where None is a variable batch size (in this prediction case it must be 1), we need to add another dimension, such that (1,9,9,2)
        #therefore prediction is an array (of batch_size), since our batch_size = 1, we need the first prediction
        #board = state.reshape(1, game_size*game_size) #reshape to (1,81) (for a basic game)
        board = state[:,:,1] #board: unfolded/folded states in (1,game_size*game_size) shape
        prediction[board==0] = np.min(prediction) #set all unfoleded to min-prediction to prevent no_progress
        #prediction[board[1]==0] = np.min(prediction)
        move = np.argmax(prediction)
        #argmax_move = tf.math.argmax(prediction[0])
        #move = np.unravel_index(argmax_move, game_instance.field.shape)
        return move, False

    def _getActions(self, game_instance: ms.Minesweeper):
        return np.invert(game_instance.field)

    def train_short_memory(self, is_game_lost):
        self.trainer.train_step(is_game_lost)

    def remember(self, state, state_new, move, reward, done):
        self.trainer.memory.append((state, state_new, move, reward, done))

    def getState(self, game_instance: ms.Minesweeper):
        result = np.zeros((game_instance.game_size, game_instance.game_size, 2))
        filter = ~np.logical_or(game_instance.field == False, game_instance.field_assignment == 0) #Not U or E
        result[filter, 0] = game_instance.field_assignment[filter]
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

def calculate_reward(game:ms.Minesweeper, done, field_already_unfolded, old_score, has_neighbors):
    reward = .3
    if field_already_unfolded:
        reward = -0.3
    elif done:
        reward = 1 if game.is_game_won() else -1
    elif old_score > 0 and not has_neighbors:
        reward = -.3

    return reward

def train():

    high_score = 0
    total_score = 0
    hundred_games_score = 0
    win_rate = 0
    plot_scores = []
    plot_mean_scores = []
    agent = Agent(load_model=True)

    game = ms.Minesweeper(GAME_SIZE, MINE_COUNT)
    progress_list, wins_list, ep_rewards = [], [], []
    random_counter = 0
    field_already_unfolded_multiplier = 1
    episodes = 100_000
    for episode in tqdm(range(1, episodes+1), unit='episode'):
        game = ms.Minesweeper(GAME_SIZE, MINE_COUNT)
        agent.number_of_games += 1
        past_n_wins = win_rate
        episode_reward = 0
        done = False
        while not done:
            state = agent.getState(game)
            move, is_random_move = agent.getAction(state, GAME_SIZE) #the next move aka field to click on
            random_counter += is_random_move

            old_score = game.unfolded
            field_already_unfolded = game.field[move[0], move[1]]
            has_unfolded_neighbours = has_neighbour(move, game)
            done = game.unfold(move[0], move[1])
            score = game.unfolded
            reward = calculate_reward(game, done, field_already_unfolded, old_score, has_unfolded_neighbours)
            episode_reward += reward

            state_new = agent.getState(game)
            agent.remember(state, state_new, move, reward, done)
            agent.train_short_memory(done)
        high_score = max(high_score, score)
        hundred_games_score += score
        ep_rewards.append(episode_reward)

        if game.is_game_won():
            win_rate += 1
            print(f"+++++++++ WIR HABEN GEWONNEN🔫🔫🔫 +++++++")

        if agent.number_of_games % 100 == 0:
            total_score += hundred_games_score
            print("Games", agent.number_of_games, "last_hundred_games_score", hundred_games_score / 100, "highscore", high_score, "average score", total_score / agent.number_of_games, "avg random move", random_counter / 100, "win-rate", win_rate/100)
            hundred_games_score = 0
            random_counter = 0
            win_rate = 0

        if win_rate > past_n_wins:
            wins_list.append(1)
        else:
            wins_list.append(0)


        if len(agent.trainer.memory) < 1000:
            continue

        if not episode % AGG_STATS_EVERY:
            win_rate = round(np.sum(wins_list[-AGG_STATS_EVERY:]) / AGG_STATS_EVERY, 2)
            med_reward = round(np.median(ep_rewards[-AGG_STATS_EVERY:]), 2)

            """agent.tensorboard.update_stats(
                progress_med = med_progress,
                winrate = win_rate,
                reward_med = med_reward,
                learn_rate = agent.learn_rate,
                epsilon = agent.epsilon)
            """
            print(f'Episode: {episode}, Median reward: {med_reward}, MEEEAN : {np.mean(ep_rewards[-AGG_STATS_EVERY:])}, Win rate : {win_rate}')

        if not episode % SAVE_MODEL_EVERY:
            with open(f'models/model_{episode}.pkl', 'wb') as output:
                pickle.dump(agent.trainer.memory, output)

            agent.trainer.model.save(f'models/model_{episode}.h5')

if __name__ == "__main__":
    train()