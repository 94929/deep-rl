import gym
import random
import numpy as np
import matplotlib.pyplot as plt

from gym.envs.registration import register

# A helper function which randomly picks an aciton between argmaxed indices
# Refered from https://gist.github.com/stober/1943451
def random_argmax(vector):
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return random.choice(indices)

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'is_slippery':False}
)

env = gym.make('FrozenLake-v3')

""" The environment looks like as below

    s0 , s1 , s2 , s3 ,
    s4 , s5 , s6 , s7 ,
    s8 , s9 , s10, s11,
    s12, s13, s14, s15,

    And each state, you can perform one of LEFT(0), DOWN(1), RIGHT(2), UP(3)

"""

nb_states = env.observation_space.n     # 16
nb_actions = env.action_space.n         # 4

# Initialise Q-Table (16x4 sized) with zeros
Q = np.zeros([nb_states, nb_actions])

# Set number of iterations
nb_episodes = 1500

# Create a list to contain cumulative rewards per episode
cumulative_rewards = []

for _ in range(nb_episodes):
    
    state = env.reset()     # Init env and receive initial state of the agent
    cumulative_reward = 0   # Set the cumulative reward to be zero
    done = False            # Set done to False, is required to enter the loop

    # Q-Table learning algorithm
    while not done:
        # Choose an action from the current state
        action = random_argmax(Q[state, :])

        # Receive a feedback after taking the action
        new_state, reward, done, _ = env.step(action)

        # Update Q-Table using the Q-Learning formula 
        Q[state, action] = reward + np.max(Q[new_state, :])

        cumulative_reward += reward
        state = new_state

    cumulative_rewards.append(cumulative_reward)

# The success rate can be calculated as below because 'if succ then reward=1 else reward=0'
succ_rate = sum(cumulative_rewards) / nb_episodes
print('Success rate:', succ_rate)
print('Final Q-Table values')
print('Left Down Right Up')
print(Q)

# Print result
plt.bar(range(len(cumulative_rewards)), cumulative_rewards, color='blue')
plt.show()

