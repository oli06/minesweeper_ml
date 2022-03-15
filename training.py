
import pickle
from tqdm import tqdm
import numpy as np

from agent import LEARNING_RATE, MAX_MEMORY, Agent
from bruteforce import BruteforceMinesweeperObject
import minesweeper as ms
from utils import has_neighbour

GAME_SIZE = 9
MINE_COUNT = 10
AGG_STATS_EVERY = 100 # calculate stats every 100 games for tensorboard
SAVE_MODEL_EVERY = 1000 # save model and replay every 10,000 episodes
EPISODES = 100_000

MAX_MEMORY = 25_000
LEARNING_RATE = 0.001
EPSILON=1
GAMMA=0.1

def train():
    high_score = 0
    total_score = 0
    hundred_games_score = 0
    win_rate = 0
    total_wins = 0
    #use model_path to use an (already) pretrained model
    agent = Agent(GAME_SIZE, LEARNING_RATE, EPSILON, GAMMA, MAX_MEMORY)
    game = ms.Minesweeper(GAME_SIZE, MINE_COUNT)
    wins_list, ep_rewards = [], []
    random_counter = 0
    for episode in tqdm(range(1, EPISODES+1), unit='episode'):
        game = ms.Minesweeper(GAME_SIZE, MINE_COUNT)
        bruteforceGameInstance = BruteforceMinesweeperObject(GAME_SIZE, GAME_SIZE)
        agent.number_of_games += 1
        past_n_wins = win_rate
        episode_reward = 0
        unfolded_multiplier = 1
        done = False
        while not done:
            state = agent.getState(game)
            move, is_random_move = agent.getAction(state, GAME_SIZE, game, bruteforceGameInstance) #the next move aka field to click on
            random_counter += is_random_move

            old_score = game.unfolded
            field_already_unfolded = game.field[move[0], move[1]]
            unfolded_multiplier = unfolded_multiplier * (1 + field_already_unfolded)
            has_unfolded_neighbours = has_neighbour(move, game)
            done = game.unfold(move[0], move[1])
            score = game.unfolded
            reward = agent.calculate_reward(game, done, field_already_unfolded, old_score, has_unfolded_neighbours, unfolded_multiplier)
            episode_reward += reward

            state_new = agent.getState(game) 
            agent.remember(state, state_new, move, reward, done)
            agent.train_short_memory(done)
        high_score = max(high_score, score)
        hundred_games_score += score
        ep_rewards.append(episode_reward)

        if game.is_game_won():
            win_rate += 1
            total_wins += 1

        if agent.number_of_games % 100 == 0:
            #with open("epsilon60.txt", "a") as out:
                #out.write(f"{win_rate},{hundred_games_score / 100},{round(np.median(ep_rewards[-AGG_STATS_EVERY:]), 2)}\n")
            total_score += hundred_games_score
            print("overall highscore", high_score, "avg. score", total_score / agent.number_of_games)
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

            print(f'Episode: {episode}, Median reward: {med_reward}, Mean reward : {np.mean(ep_rewards[-AGG_STATS_EVERY:])}, Win rate : {win_rate}, total won games: {total_wins}')

        if not episode % SAVE_MODEL_EVERY:
            with open(f'models/model_{episode}.pkl', 'w+b') as output:
                pickle.dump(agent.trainer.memory, output)

            agent.trainer.model.save(f'models/model_{episode}.h5')

if __name__ == "__main__":
    train()