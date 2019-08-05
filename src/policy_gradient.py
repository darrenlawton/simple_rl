# Reference: https://github.com/rlcode/reinforcement-learning/blob/master/2-cartpole/3-reinforce/cartpole_reinforce.py

import sys
import gym
import pylab
import warnings
import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.95
N_EPISODES = 1000


class pg_agent:
    def __init__(self, state_space, action_space):
        self.state_space= state_space
        self.action_space = action_space

        # build policy network
        self.model = self.build_model()

        # lists for the states, actions and rewards
        self.states, self.actions, self.rewards = [], [], []

    def one_hot_state(self, state):
        state_m = np.zeros((1, self.state_space))
        state_m[0][state] = 1
        return state_m

    # NN to approximate policy.
    # state is input, and p_a is output
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_space, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(24, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(self.action_space, activation='softmax', kernel_initializer='glorot_uniform'))
        model.summary()

        model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=LEARNING_RATE))
        return model

    # Stochastically pick action from output of policy network
    def get_action(self, state):
        policy = self.model.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_space, 1, p=policy)[0]

    # Sample returns are used to evaluate policy
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        summation = 0
        for i in reversed(range(0, len(rewards))):
            summation = summation * DISCOUNT_FACTOR + rewards[i]
            discounted_rewards[i] = summation

        return discounted_rewards

    # Save tuple of each step
    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)

    # Update policy network after each episode
    def train_model(self):
        episode_length = len(self.states)
        # normalise discounted rewards
        discounted_rewards = self.discount_rewards(self.rewards)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        update_inputs = np.zeros((episode_length, self.state_space))
        advantages = np.zeros((episode_length, self.action_space))

        for i in range(episode_length):
            update_inputs[i] = self.states[i]
            advantages[i][self.actions[i]] = discounted_rewards[i]

        self.model.fit(update_inputs, advantages, epochs=1, verbose=0)
        self.states, self.actions, self.rewards = [], [], []


def main():
    env = gym.make('FrozenLake-v0')

    state_space = env.observation_space.n
    action_space = env.action_space.n

    agent = pg_agent(state_space, action_space)

    scores, episodes = [], []

    for e in range(N_EPISODES):
        done = False
        step = 0
        state = env.reset()
        state = agent.one_hot_state(state)

        while not done:
            # get action for current state
            action = agent.get_action(state)
            state_next, reward, done, _ = env.step(action)
            state_next = agent.one_hot_state(state_next)
            if done: reward = 1
            step += 1
            agent.append_sample(state, action, reward)

            state = state_next

            if done:
                agent.train_model()
                scores.append(step)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/frozen_lake_reinforce.png")
                print("episode:", e, "  steps:", step)

                if np.mean(scores[-min(10, len(scores)):]) > 100:
                    sys.exit()

    print("Score over time: " + str(sum(scores) / N_EPISODES))


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()
