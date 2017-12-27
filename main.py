import gym
import numpy as np
import matplotlib.pyplot as plt

# Define our world to be non-deterministic(i.e. stochastic), 'is_slippery=True'
env = gym.make('FrozenLake-v0')

# Initialise Q-Table with zeros representing rewards
nb_states = env.observation_space.n
nb_actions = env.action_space.n
Q = np.zeros([nb_states, nb_actions])

# Set learning rate, alpha representing how much the agent listen to Q-Table
alpha = .85

# Set discount reward factor, gamma
gamma = .99

# Set number of iterations
nb_episodes = 1500

# Create a list to contain cumulative rewards per episode
cumulative_rewards = []

for i in range(nb_episodes):
    
    state = env.reset()     # Init env and receive initial state of the agent
    cumulative_reward = 0   # Set the cumulative reward to be zero
    done = False            # Set done to False, is required to enter the loop

    # The Q-Table learning algorithm with discounted reward
    while not done:
        # Choose an action which maximises the reward (from the current state)
        # When choosing an action, use decaying random noise for path exploration
        action = np.argmax(Q[state, :] + np.random.randn(1, nb_actions) / (i + 1))

        # Receive a feedback after taking the action
        new_state, reward, done, _ = env.step(action)

        # In stochastic world, we need not to rely on previous Q-Table
        # This can be done by adding the learning rate variable, alpha
        # Q(s, a) = (1-alpha)*Q(s, a) + alpha*(reward + gamma*max(a').Q(s', a')
        Q[state, action] = (1 - alpha) * Q[state, action] \
                            + alpha * (reward + gamma * np.max(Q[new_state, :]))

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

