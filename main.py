import gym
import random
import numpy as np
import matplotlib.pyplot as plt

from gym.envs.registration import register

# Environment set up
register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'is_slippery':False}
)

env = gym.make('FrozenLake-v3')

# Set parameters for Q-Table
nb_states = env.observation_space.n     # 16
nb_actions = env.action_space.n         # 4

# Initialise Q-Table (16x4 sized) with zeros
Q = np.zeros([nb_states, nb_actions])

# Set discount reward factor, gamma
gamma = 0.9

# Set number of iterations
nb_episodes = 1500

# Create a list to contain cumulative rewards per episode
cumulative_rewards = []

for i in range(nb_episodes):
    
    state = env.reset()     # Init env and receive initial state of the agent
    cumulative_reward = 0   # Set the cumulative reward to be zero
    done = False            # Set done to False, is required to enter the loop

    e = 1. / ((i//100)+1)   # Set decaying e value for our algorithm

    """ Why do we do 'i // 100'?

    # Setting reasonable e is crucial because it will affect agent's preference.
    # If e is too small, it tends to exploit (a lot) rather than explorate env.
    # Meaning that it won't know what to do when rewards from actions are the same.
    # Same problem as before, of 'action = np.argmax(Q[state, :])'
    
    """

    # The Q-Table learning algorithm with discounted reward
    # Q(s, a) = reward + gamma * max.Q'(s', a')
    while not done:
        # Choose an action which maximises the reward (from the current state)
        # When choosing an action, use decaying e-greedy algorithm
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        # Receive a feedback after taking the action
        new_state, reward, done, _ = env.step(action)

        # Update Q-Table using the algorithm with discounted reward factor
        # Q(s, a) = reward + gamma * max(a').Q'(s', a')
        Q[state, action] = reward + gamma * np.max(Q[new_state, :])

        cumulative_reward += reward
        state = new_state

    cumulative_rewards.append(cumulative_reward)

# Print the result
succ_rate = sum(cumulative_rewards) / nb_episodes
print('Success rate:', succ_rate)
print('Final Q-Table values')
print('Left Down Right Up')
print(Q)

# Plot the result
plt.bar(range(len(cumulative_rewards)), cumulative_rewards, color='blue')
plt.show()

