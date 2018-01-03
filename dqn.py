import tensorflow as tf
import numpy as np

from config import learning_rate
from config import discount_factor

""" DQN (NIPS 2013 Version)

    Issues with Traditional Qnet
    
    1. Correlations between samples (when training with few nb_elements)
    2. Non-stationary targets (because only one network is used for both Qpred and Y)

    Solutions
    
    0. Go Deep (create a deeper network in terms of nb_hidden_layers)
    1. Capture and Play (create a buffer which stores histories of play and do random sampling)
    2. Separate networks (create a non-stationary target network, done in Nature 2015)

"""
class dqn:
    
    # Initialize hyper-parameters required to build the network
    def __init__(self, session, input_size, output_size, name='main'):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.network_name = name
        self.build()

    # Build and initialize this DQN
    def build(self):
        with tf.variable_scope(self.network_name):
            # Define input layer, X
            self.X = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32)

            # The first hidden layer of weights
            W1 = tf.get_variable('W1', shape=[self.input_size, 10],
                                 initializer=tf.contrib.layers.xavier_initializer())
            L1 = tf.tanh(tf.matmul(self.X, W1))

            # The second hidden layer of weights
            W2 = tf.get_variable('W2', shape=[10, 10],
                                 initializer=tf.contrib.layers.xavier_initializer())
            L2 = tf.tanh(tf.matmul(L1, W2))

            # The third hidden layer of weights
            W3 = tf.get_variable('W3', shape=[10, self.output_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            L3 = tf.tanh(tf.matmul(L2, W3))

            # Define output layer, Qpred
            self.Qpred = L3

            # Define our expectation, Y
            self.Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)

            # Loss function
            self.loss = tf.reduce_mean(tf.square(self.Y - self.Qpred))

            # Training using optimizer
            self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    # Predict the Q-Values (i.e. rewards of actions) from a given state
    def predict(self, state):
        # Preprocess the input state
        x = np.reshape(state, [1, self.input_size])

        # Predict the rewards from the pre-processed state
        return self.session.run(self.Qpred, feed_dict={self.X:x})

    # Train (i.e. update) DQN using the batch data from the replay buffer
    def train(self, batch):
        x_stack = np.empty(0).reshape(0, self.input_size)
        y_stack = np.empty(0).reshape(0, self.output_size)

        for state, action, reward, next_state, done in batch:
            Q = self.predict(state)

            if done:
                Q[0, action] = reward
            else:
                Q[0, action] = reward + discount_factor*np.max(self.predict(next_state))

            x_stack = np.vstack([x_stack, state])
            y_stack = np.vstack([y_stack, Q])
        
        #print('x', x_stack.shape)
        #print('y', y_stack.shape)

        return self.session.run([self.loss, self.trainer],
                                feed_dict={self.X:x_stack, self.Y:y_stack})

    # Run the algorithm using our trained DQN in a given game environment
    def run(self, env):
        
        state = env.reset()
        step_count = 0
        while True:
            step_count += 1
            env.render()
            action = np.argmax(self.predict(state))
            state, reward, done, _ = env.step(action)
            
            if done:
                print('nb_steps:', step_count)
                break

