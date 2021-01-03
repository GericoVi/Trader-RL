import gym, gym_anytrading
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK
import matplotlib.pyplot as plt
import os
import quantstats as qs
import pandas as pd

import numpy as np 
import tensorflow as tf
from tensorflow import keras

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

DF = FOREX_EURUSD_1H_ASK.copy()
WINDOW_SIZE = 10
start_index = len(DF) - 2200 # ~3 months
end_index = len(DF)

MODEL_PATH = "temp_models\\Tod-OnlyClose-10window-last3months_episode150.model"

# Initiliase environment
env = gym.make('forex-v0', df = DF, frame_bound=(start_index, end_index), window_size=WINDOW_SIZE)

# Load model
agent = keras.models.load_model(MODEL_PATH)
agent.summary()

# Simulate environment with agent actions - no training
state = env.reset()
episode_reward = 0
done = False

while not done:
    # Get action from net
    action = np.argmax(agent.predict(np.array(state).reshape(-1, np.shape(state)[0],np.shape(state)[1]))[0])

    # Take action and update environment
    new_state, reward, done, _ = env.step(action)
    #env.render()

    # Update episode reward and state
    episode_reward += reward
    state = new_state

plt.cla()
env.render_all()
plt.show()

# Analyse trading strategy with quantstats
qs.extend_pandas()

net_worth = pd.Series(env.history['total_profit'], index=FOREX_EURUSD_1H_ASK.index[start_index+1:end_index])
returns = net_worth.pct_change().iloc[1:]

qs.reports.full(returns)
qs.reports.html(returns, output='results/test.html')
