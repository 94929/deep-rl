import gym
import numpy as np
import tensorflow as tf

# Define our world to be 'CartPole'
env = gym.make('CartPole-v0')

# Set input and output size used for Q-Network
input_size  = env.observation_space.shape[0] # (4,)
output_size = env.action_space.n             # 2

# Set learning rate, alpha
alpha = 1e-1

# Set discount (reward) factor, gamma
gamma = .99

# Set number of iterations
nb_episodes = 1500

""" The Q-Network 

    X = Input layer
    W = The first Hidden layer

    Qpred = Predicted value
    Y = Expected value

    loss = diff = sum $ square $ Q - Y
    train = GDO(learning_rate).minimize(loss)
"""

# #############################################################################

# Define layer parameters
X = tf.placeholder(shape=[None, input_size], dtype=tf.float32)

# First layer
W1 = tf.get_variable(name='W1', shape=[input_size, output_size], \
                    initializer=tf.contrib.layers.xavier_initializer())

Qpred = tf.matmul(X, W1)
Y = tf.placeholder(shape=[None, output_size], dtype=tf.float32)

# Now, calculaing the loss, it's matrix subtraction, use square and reduce_sum of tf
loss = diff = tf.reduce_sum(tf.square(Qpred - Y))

# Train using GradientDescentOptimizer with given learning_rate and loss obtained above
train = tf.train.AdamOptimizer(learning_rate=alpha).minimize(loss)

# #############################################################################

results = []
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    for i in range(nb_episodes):
    
        state = env.reset()         
        epsilon = .1 / ((i//50)+1)  
        done = False                
        
        # Set step_count used for checking our network
        step_count = 0 

        # The Q-Network learning (training) algorithm 
        while not done:
            step_count += 1

            # (1) Choose an action with epsilon greedy algorithm
            if np.random.rand(1) < epsilon:
                # Take an action randomly for exloration purpose
                action = env.action_space.sample()
            else:
                # Preprocess(i.e. reshpae) the input, state
                x = np.reshape(state, [1, input_size])

                # By using our network defined Qpred,
                # Obtain Q values (i.e. rewards) when the agent is at state s
                Q = sess.run(Qpred, feed_dict={X:x})

                # Choose action which maximises the reward
                action = np.argmax(Q)

            # (2) Update the reward into Q-Network
            next_state, reward, done, _ = env.step(action)
            if done:
                # In env, we don't want the game to be done, if done give penalty
                Q[0, action] = -100
            else:
                next_x = np.reshape(next_state, [1, input_size])
                next_Q = sess.run(Qpred, feed_dict={X:next_x})

                Q[0, action] = reward + gamma*np.max(next_Q)

            # Train our network using target (Y) and predicted (Qpred) values
            # i.e. Telling the network that when input:x we expect output:Q
            # the network will adjust its values in the network according to the above
            sess.run(train, feed_dict={X:x, Y:Q})

            state = next_state
    
        results.append(step_count)
        print('Episode: {}, steps: {}'.format(i, step_count))
        
        # Finish training if the result seems reasonably good enough
        # i.e. the last five episodes finished with at least 500 steps
        if len(results) > 5 and np.mean(results[-5:]) > 500:
            break

    # After training, test our network in action
    observation = env.reset()
    done = False
    while not done:
        x = np.reshape(observation, [1, input_size])
        Qs = sess.run(Qpred, feed_dict={X:x})
        action = np.argmax(Qs)

        observation, _, done, _ = env.step(action)

