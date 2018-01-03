import gym
import numpy as np
import tensorflow as tf
import collections
import random

from dqn import dqn

# Create and define our new environment, CartPole Game
env = gym.make('CartPole-v0')
env._max_episode_steps = 10000

# Set hyper-parameters required for dqn algorithm
input_size  = env.observation_space.shape[0] # (4,)
output_size = env.action_space.n             # 2

nb_episodes = 1000  # number of episodes
gamma = .99         # discount factor

REPLAY_MEMORY = 1000
replay_buffer = collections.deque(maxlen=REPLAY_MEMORY)

with tf.Session() as sess:
    
    main_dqn = dqn(sess, input_size, output_size)
    sess.run(tf.global_variables_initializer())
    for episode in range(nb_episodes):
    
        state = env.reset()
        epsilon = .1 / ((episode // 10) + 1)
        done = False
        step_count = 0 

        # The DQN learning (training) algorithm 
        while not done and step_count < 5000:

            # (1) Choose an action with e-greedy algorithm
            if np.random.rand(1) < epsilon:
                action = env.action_space.sample()
            else:
                # Use main_dqn to predict rewards of actions from a given state
                action = np.argmax(main_dqn.predict(state))

            # (2) Update the reward into Q-Network
            next_state, reward, done, _ = env.step(action)
            if done:
                # Edit reward if the agent died
                reward = -100
            
            # Save current play history into the buffer
            current_play = (state, action, reward, next_state, done)
            replay_buffer.append(current_play)

            state = next_state
            step_count += 1
    
        # Print the result of the current episode
        print('Episode: {}, steps: {}'.format(episode, step_count))

        # Train DQN every 10 episodes
        if episode >= 10 and episode % 10 == 0:
            # Train for 100 iteration
            for _ in range(100):
                batch = random.sample(replay_buffer, 10)
                loss, _ = main_dqn.train(batch)
                print('Current DQN Loss:', loss)

    # Try to run the game after training the network
    main_dqn.run(env)

