import gym
import numpy as np
import random
import warnings
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.95

MEMORY_SIZE = 1000
BATCH_SIZE = 25
N_EPISODES = 500

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995


class DQN:

    def __init__(self, state_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.state_space = state_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(state_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def one_hot_state(self, state):
        state_m = np.zeros((1, self.state_space))
        state_m[0][state] = 1
        return state_m

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return None

        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in batch:
            q_update = reward
            if not done:
                q_update = (reward + DISCOUNT_FACTOR * np.amax(self.model.predict(next_state)[0]))

            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def main():
    env = gym.make('FrozenLake-v0')
    state_space = env.observation_space.n
    action_space = env.action_space.n
    dqn = DQN(state_space, action_space)
    reward_list = []

    for i in range(N_EPISODES):
        state = env.reset()
        state = dqn.one_hot_state(state)
        step = 0

        while True:
            step += 1
            action = dqn.act(state)
            state_next, reward, done, _ = env.step(action)
            state_next = dqn.one_hot_state(state_next)
            dqn.remember(state, action, reward, state_next, done)
            state = state_next

            if done:
                print("Run: " + str(i) + ", exploration: " + str(dqn.exploration_rate) + ", score: " + str(step))
                reward_list.append(step)
                break
            dqn.experience_replay()

    print("Score over time: " + str(sum(reward_list) / N_EPISODES))
    plt.plot(reward_list)
    plt.show()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()
