import os, random, time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

import gym
import gym_anytrading
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK

from Agents.agents import DQNAgent

# Silence tensorflow info and warning prints
'''
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

PATH = ""
GRID_RESULTS_FILE = "test"

# Simulation settings
EPISODES = 2
AGGREGATE_STATS_EVERY = 50
MIN_REWARD = -200

# Exploration settings
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

# Env settings
WINDOW_SIZE = 10
DF = FOREX_EURUSD_1H_ASK.copy()

# Iterate through episodes
# agent & env - objects
# episode = num episodes to run | eps_decay = (minimum epsilon, epsilon decay) | aggregate_stats_every = aggregate stats every n episodes for tensorboard
# show_every = either false or render episode for every n episodes
# save_while_training = flag to save model and weights during training when it reaches a minimum average reward (specified by min_reward_for_saving)
def simulate_episodes(agent, env, episodes, eps_decay, aggregate_stats_every, show_every=False, save_while_training=False, min_reward_for_saving=0):
    epsilon = 1
    ep_rewards = np.array([-200])

    for episode in tqdm(range(agent.episodes_done+1, agent.episodes_done+episodes+1), ascii=True, unit='episode'):
        # Reset episode reward and step num
        episode_reward = 0
        step = 1

        # Reset env
        state = env.reset()

        # Reset done flag and interate through steps until episode ends
        done = False
        while not done:
            # Exploitation vs exploration
            if np.random.random() > epsilon:
                # Do 'best' action - from pov of model
                action = np.argmax(agent.model.predict(np.array(state).reshape(-1, agent.input_shape[0],agent.input_shape[1]))[0])
            else:
                # Do random action
                action = np.random.randint(0, env.action_space.n)

            # Take action and update environment
            new_state, reward, done, _ = env.step(action)

            # Update episode reward
            episode_reward += reward

            # Add to replay memory then train 'main' network
            agent.update_replay_memory((state, action, reward, new_state, done))
            agent.train(done, step)
            state = new_state
            step += 1

        # Append episode reward and log stats when appropriate
        np.append(ep_rewards, episode_reward)
        if episode % aggregate_stats_every == 0 or episode == 1:
            average_reward = np.mean(ep_rewards[-aggregate_stats_every:])
            min_reward = np.min(ep_rewards[-aggregate_stats_every:])
            max_reward = np.max(ep_rewards[-aggregate_stats_every:])

            # Save model if min reward reached and if flag
            if average_reward >= min_reward_for_saving and save_while_training:
                agent.model.save(f'{PATH}temp_models\\{agent.name}_episode{episode}.model')

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    return episode, max_reward, min_reward, average_reward

###################################################################################################################################################
###################################################################################################################################################

# Initialise environment
#start_index = WINDOW_SIZE
start_index = len(DF) - 2200 # ~3 months
end_index = len(DF)
env = gym.make('forex-v0', df = DF, frame_bound=(start_index, end_index), window_size=WINDOW_SIZE)
obs = env.reset()

# Specify conv net configuration
'''
models_config = [ {"name": "one", "conv_list": [[128, 1, 0.2]], "dense_list": [32], "minibatch_size": 64},
                  {"name": "two", "conv_list": [[128, 1, 0.2], [128, 1, 0.2]], "dense_list": [32], "minibatch_size": 64},
                  {"name": "three", "conv_list": [[128, 1, 0.2]], "dense_list": [32, 32], "minibatch_size": 64},
                  {"name": "four", "conv_list": [[128, 1, 0.2], [128, 1, 0.2]], "dense_list": [32, 32], "minibatch_size": 64},
                  {"name": "five", "conv_list": [[128, 1, 0.2]], "dense_list": [32], "minibatch_size": 32},
                  {"name": "six", "conv_list": [[128, 1, 0.2], [128, 1, 0.2]], "dense_list": [32], "minibatch_size": 32} ]
'''

models_config = [ {"name": "Tod-OnlyClose-10window-last3months", "conv_list": [[128, 1, 0.2]], "dense_list": [32], "minibatch_size": 32} ]

# Dataframe to store results of grid search
results = pd.DataFrame(columns = ["Model Name", "Convolutional Layers", "Dense Layers", "Minibatch Size", "Average Reward", "Max Reward", "Min Reward"])

# Set RNG seeds
random.seed(123)
np.random.seed(123)
tf.compat.v1.set_random_seed(123)

# Create temp models folder
if not os.path.isdir('temp_models'):
    os.makedirs('temp_models')

# Create trained models folder
if not os.path.isdir('models'):
    os.makedirs('models')

for model in models_config:
    startTime = time.time()
    # Initialise agent
    agent = DQNAgent(model["name"], np.shape(obs), (env.action_space.n), model["conv_list"], model["dense_list"], model["minibatch_size"])
    #agent.continue_training("models\\Tod-OnlyClose-10window-last3months_Eps100_max-200.00_avg-200.00_min-200.00.model", 100)

    # Show model architecture
    agent.model.summary()

    # Iterate over episodes - Engage with environment - and use progress bar
    episode, max_reward, min_reward, average_reward = simulate_episodes(agent, env, EPISODES, (MIN_EPSILON, EPSILON_DECAY), AGGREGATE_STATS_EVERY, show_every=AGGREGATE_STATS_EVERY, save_while_training=True, min_reward_for_saving=MIN_REWARD)

    # Save trained model
    agent.model.save(f"{PATH}models/{agent.name}_Eps{episode}_max{max_reward:_>7.2f}_avg{average_reward:_>7.2f}_min{min_reward:_>7.2f}.model")

    print()
    print(f"Training took {round(time.time() - startTime)/60} minutes")

    results = results.append( {"Model Name": model["name"], "Convolutional Layers": model["conv_list"], "Dense Layers": model["dense_list"], "Minibatch Size": model["minibatch_size"], "Average Reward": average_reward, "Max Reward": max_reward, "Min Reward": min_reward}, 
                                ignore_index = True)

# Create results folder
if not os.path.isdir('results'):
    os.makedirs('results')

results.to_csv(f"{PATH}results/{GRID_RESULTS_FILE}.csv", index=False)