# Reference: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

import numpy as np
import gym

env = gym.make('FrozenLake-v0')

q_table = np.zeros([env.observation_space.n,env.action_space.n])
learning_rate = 0.95
discount_rate = 0.95
n_episodes = 5000
reward_list = []

for i in range(n_episodes):
    state = env.reset()
    all_rewards = 0
    goal_state = False
    step = 0

    while True:
        step += 1
        # choose action greedily (with noise)
        action = np.argmax(q_table[state,:] + np.random.randn(1, env.action_space.n)*(1./(i+1)))
        # given action, get new state and reward
        state_1, reward, goal_state, _ = env.step(action)
        # update q-table with new knowledge
        q_table[state, action] = q_table[state, action] + learning_rate * ((reward + discount_rate *
                                                                           np.max(q_table[state_1,:])) -
                                                                           q_table[state, action])

        all_rewards += reward
        state = state_1
        if goal_state:
            print("{0} goal square reached in {1} steps." .format(i, step))
            reward_list.append(step)
            break



print ("Score over time: " +  str(sum(reward_list)/n_episodes))
print("Q Values: {} " .format(q_table))



