import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Define our world to be non-deterministic(i.e. stochastic), 'is_slippery=True'
env = gym.make('FrozenLake-v0')

# Set input and output size used for Q-Network
input_size  = env.observation_space.n
output_size = env.action_space.n

def one_hot(x):
    return np.identity(input_size)[x : x+1]

# Set learning rate, alpha
alpha = .1

# Set discount (reward) factor, gamma
gamma = .99

# Set number of iterations
nb_episodes = 1500

""" The Q-Network 

    The network inputs any state from 0 to 15.
    The network outputs an action from 0 to 3.

    X = Input layer, 1x16 sized matrix
    W = The first Hidden layer, 16x4 sized matrix

    Qpred = Predicted value, 1x4 sized matrix
    Y = Expected value

    loss = diff = sum $ square $ Q - Y
    train = GDO(learning_rate).minimize(loss)
"""

# #############################################################################

# Define layer parameters
X = S = tf.placeholder(shape=[1, input_size], dtype=tf.float32)
W = tf.Variable(tf.random_uniform(shape=[input_size, output_size], minval=0, maxval=0.01))

Qpred = tf.matmul(X, W)
Y = tf.placeholder(shape=[1, output_size], dtype=tf.float32)

# Now, calculaing the loss, it's matrix subtraction, use square and reduce_sum of tf
loss = diff = tf.reduce_sum(tf.square(Qpred - Y))

# Train using GradientDescentOptimizer with given learning_rate and loss obtained above
train = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(loss)

# #############################################################################

# Create a list to contain cumulative rewards per episode
cumulative_rewards = []

# Create new Session, sess
with tf.Session() as sess:
    
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(nb_episodes):
    
        state = env.reset()     # Reset env and get the initial observation
        epsilon = .1 / (i+1)    # Use epsilon greedy algorithm for exploration
        cumulative_reward = 0   # Set the cumulative reward to be zero
        done = False            # Set done to False, is required to enter the loop

        # The Q-Network learning (training) algorithm is splited into two parts
        # (1) Choose an action at each step, (2) Save the reward derived
        while not done:

            # (1) Choose an action
            # The line below predicts Q values from a given state
            # The Q contains expected rewards of each action from given state
            # e.g. Q = [[ 0.00420892  0.00085265  0.00160731  0.49378362]]
            Q = sess.run(Qpred, feed_dict={X:one_hot(state)})
            #print('Qs value: ', Q)
            if np.random.rand(1) < epsilon:
                # Take an action randomly for exloration purpose
                action = env.action_space.sample()
            else:
                # Take an action which yields the best reward according to Q
                action = np.argmax(Q)

            # After taking an action, get new state and reward
            new_state, reward, done, _ = env.step(action)

            # (2) Save the reward, it's done according to the equation below
            # reward if terminal_state else reward + gamma*max(Q)
            if done:
                # Q[0, _] is crucial since Qpred is 1x4 matrix of [[a1, a2 ,a3, a4]]
                Q[0, action] = reward
            else:
                # Obtain new Q values by feeding the new state through our network
                # Same as for the above Q obtained, but this is Q for new state
                Qtmp = sess.run(Qpred, feed_dict={X:one_hot(new_state)})
                #print('Qtmp: ', Qtmp)
                # Update Q
                Q[0, action] = reward + gamma*np.max(Qtmp)

            # 'Train' our network using target (Y) and predicted (Qpred) values
            sess.run(train, feed_dict={X:one_hot(state), Y:Q})

            cumulative_reward += reward
            state = new_state
    
        cumulative_rewards.append(cumulative_reward)

# Print the result
succ_rate = sum(cumulative_rewards) / nb_episodes
print('Success rate:', succ_rate)

# Plot the result
plt.bar(range(len(cumulative_rewards)), cumulative_rewards, color='blue')
plt.show()

