import random
from collections import deque
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam

# Agent settings
REPLAY_MEMORY_SIZE = 5000 
MIN_REPLAY_MEMORY_SIZE = 1000
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5

# Deep q-learning network agent
class DQNAgent:
    def __init__(self, name, input_shape, output_shape, conv_list, dense_list, minibatch_size):
        # Store network properties
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.conv_list = conv_list
        self.dense_list = dense_list
        self.name = name

        # Create the main model, trained every step
        self.model = self.create_model()

        # Target model - only trained every n episodes - initialise as the same as main model
        # The one that produces the 'future' Q values' - more stable
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # Initialise replay memory array - model will be trained from a random sample of this - deque array for faster append and pop
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # For counting when to update target network with main network's weights
        self.target_model_counter = 0

        # Size of batch to train network with
        self.minibatch_size = minibatch_size

        # For continuing training
        self.episodes_done = 0

    # For continuing training from a saved model
    def continue_training(self, model_path, episodes_previously_done):
        # Get model
        checkpoint = keras.models.load_model(model_path)

        '''
        # Get number of episodes alread done
        if model_path.find('Eps') != -1:
            index = model_path.find('Eps')
            end_idx = model_path.find('_')
        elif model_path.find('episode') != -1:
            index = model_path.find('episode')
            end_idx = model_path.find('.model')

        self.episodes_done = int(model_path[index+3:end_idx])
        '''
        self.episodes_done = episodes_previously_done

        # Load in weights
        self.model.set_weights(checkpoint.get_weights())
        self.target_model.set_weights(checkpoint.get_weights())

    # Define a 1D convolutional neural network for time series data
    # Conv_list = [[num filters, usePoolOrNot, dropout], ...] | Dense_list = [units, ..]
    def create_model(self):
        # Initialise a sequential model with keras framework
        model = Sequential()

        # Create specified number of convolutional blocks with given configuration
        for conv in self.conv_list:
            model.add(Conv1D(conv[0], 3, input_shape=self.input_shape, activation='relu'))
            if conv[1]:
                model.add(MaxPooling1D(pool_size=2))
            # Technically can just pass conv[2] as a dropout arguement and if it's 0, no units will be dropped - 
            # but might be better training and predicting performance if there isn't an unneccessary dropout layer present 
            if conv[2] > 0:
                model.add(Dropout(conv[2]))

        # Flatten output of convolutions to be fed into fully connected layers
        model.add(Flatten())

        # Create specified fully connected layers
        for units in self.dense_list:
            model.add(Dense(units, activation='relu'))

        # Fully connected output layer
        model.add(Dense(self.output_shape, activation='linear'))

        # Configure model for training
        model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics= ['accuracy'])
        
        return model

    # Add current step data to memory replay array
    # transition = (observation, action, reward, new observation, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Training function, called every step during episode
    def train(self, terminal_state, step):
        # Start training only if certain number of samples already saved in memory array
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get random saple from memory
        minibatch = random.sample(self.replay_memory, self.minibatch_size)

        # Get states of samples then get Q value from 'main' network for these states
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get states of sample after the action has been taken then query 'target' net for Q values
        new_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_states)

        # Initialise training data
        X = []
        y = []

        # Enumerate batch and loop through to create training data
        for index, (state, action, reward, _, done) in enumerate(minibatch):
            # Get new q - for deep q learning, only need part of original Q-learning equation
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update q value for state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            
            # Append to training data
            X.append(state)
            y.append(current_qs)

        # Fit model to the samples (as one batch)
        self.model.fit(np.array(X), np.array(y), batch_size=self.minibatch_size, verbose=0, shuffle=False)

        # Update target network counter every episode
        if terminal_state:
            self.target_model_counter += 1

        # Update the target network weights if the counter reaches set value (and reset counter)
        if self.target_model_counter == UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_model_counter = 0