import gym

env = gym.make('FrozenLake-v0') # Create the environment, env of FrozenLake Game
env.reset()                     # Reset env to initial observation(state)

while True:
    # For each iteration, render(display) current observation of env
    env.render()

    # The action taken by the agent
    action = env.action_space.sample()

    # state(i.e. observation) (object): representing your observation of the environment
    # reward (float): amount of reward achieved by the previous action
    # done (bool): indicates whether it's time to reset the environment again
    # info (dict): diagnostic information useful for debugging
    state, reward, done, info = env.step(action)

    # Performing an Action will result State and Reward
    print('Action: {}, State: {}, Reward: {}'.format(action, state, reward))

    if done:
        print('Finished with reward', reward)
        break

