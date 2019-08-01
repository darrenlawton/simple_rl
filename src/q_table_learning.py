import numpy as np
import gym

env = gym.make('FrozenLake-v0')

q_table = np.zeros([env.observation_space.n,env.action_space.n])
learning_rate = 0.8
discount_rate = 0.95
n_episodes = 2000

reward_list = []

for i in range(n_episodes):
    state = env.reset()
    all_rewards = 0
    end_state = False
    step = 0

    while step < 99:
        step += 1
        # choose action greedily (with noise)
        action = np.argmax(q_table[state,:] + np.random.randn(1, env.action_space.n)*(1./(i+1)))
        # given action, get new state and reward
        state_1, reward, end_state, _ = env.step(action)
        # update q-table with new knowledge
        q_table[state, action]




