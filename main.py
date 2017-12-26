import gym

env = gym.make('FrozenLake-v0') # Create the environment, env of FrozenLake Game
env.reset()                     # Reset env to initial observation(state)

action_dict = {0:'LEFT', 1:'DOWN', 2:'RIGHT', 3:'UP'}

while True:
    # Agent takes the following action
    action = env.action_space.sample()

    # After each action, env renders(display) the current observation
    env.render()

    # state(i.e. observation) (object): representing your observation of the environment
    # reward (float): amount of reward achieved by the previous action
    # done (bool): indicates whether it's time to reset the environment again
    # info (dict): diagnostic information useful for debugging
    state, reward, done, info = env.step(action)

    # Performing an Action will result State and Reward
    print('Action: {}, State: {}, Reward: {}'.format(action_dict[action], state, reward))

    if done:
        print('Finished with reward', reward)
        break

