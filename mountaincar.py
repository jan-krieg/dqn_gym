"""Deep Q-network implementation for the OpenAI Gym MountainCar-v0 environment.

We use a feedforward neural network with two fully-connected hidden layers
that consist of 128 neurons each. The model is trained via experience replay
by sampling randomly from the memory. The target Q function is given by
a second neural network with identical properties, whose weights are regularly
updated by the main model. The mean reward after training is -99.

To speed up the learning, we initially fill the memory by taking random
actions. This gives the agent information about the position of the goal.

Due to the two-dimensional state space (position and velocity), we can
visualize the neural network while it learns. We do this by discretizing
the state space and plotting the prediction of our model for max_a Q(s,a)
as well as for the greedy policy argmax_a Q(s,a).
"""

import random
import numpy
import math
import gym
import pickle

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import StrMethodFormatter

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

# -------------------- PARAMETERS ----------------------
GYM_ENV = 'MountainCar-v0'  # OpenAI Gym environment

MEMORY_CAPACITY = 2e6  # Maximum number of samples in memory
MEMORY_CAPACITY_RANDOM = 2e5  # Number of samples from random agent to initialize memory
BATCH_SIZE = 32  # Batch size for memory replay and for model training

GAMMA = 0.99  # Discount rate
LEARNING_RATE = 0.0001
UPDATE_TARGET_FREQUENCY = 1000  # Step rate for copying weights from the active model to the target model

MAX_EPSILON = 0.1  # Initial probability for random decision of the agent
MIN_EPSILON = 0.01  # Asymptotic probability for random decision of the agent
LAMBDA = 1e-4  # Rate for exponential decay of epsilon

PLOT = True  # Visualize the predictions of the neural network
RES = 64  # Plot resolution
UPDATE_PLOT_FREQUENCY = 1000  # Step rate for updating plots

RENDER = False  # Render OpenAI Gym environment
OBSERVE = True  # Add new experiencies to the memory
TRAIN = True  # Train the network via experience replay

LOAD_MEMORY = False
FILENAME_LOAD_MEMORY = "_memory_mountaincar.p"
SAVE_MEMORY = False
FILENAME_SAVE_MEMORY = "_memory_mountaincar.p"
LOAD_BRAIN = False
FILENAME_LOAD_BRAIN = "_brain_mountaincar.h5"
SAVE_BRAIN = False
FILENAME_SAVE_BRAIN = "_brain_mountaincar.h5"

# ------------------- PLOT UTILITIES -------------------
def map_brain(brain, observation_space, res=RES):
    """
    Calculate the preciditions of the neural network for the discretized state space

    :param brain: Neural network
    :param observation_space: Observation space of the OpenAI Gym environment
    :param res: Resolution for state space discretization
    :returns: Numpy arrays for max_a Q(s,a) and for argmax_a Q(s,a)
    """
    s = numpy.zeros((res * res, 2))
    i = 0

    low = observation_space.low
    high = observation_space.high

    for i1 in range(res):
        for i2 in range(res):
            s[i] = numpy.array([low[0] + (high[0]-low[0]) * i1 / (res - 1),
                                low[1] + (high[1]-low[1]) * i2 / (res - 1)])
            i += 1

    map_v = numpy.amax(brain.predict(s), axis=1).reshape((res, res))
    map_a = numpy.argmax(brain.predict(s), axis=1).reshape((res, res))

    return map_v, map_a

def initialize_plot(observation_space, res=RES):
    """
    Create two subplots for later visualization of the neural network predictions

    :param observation_space: Observation space of the OpenAI Gym environment
    :param res: Resolution for state space discretization
    :returns: Two imshow objects for max_a Q(s,a) and for argmax_a Q(s,a)
    """
    low = observation_space.low
    high = observation_space.high

    fig = plt.figure(figsize=(5, 7))

    fig.add_subplot(211)

    plot1 = plt.imshow(numpy.random.rand(res,res), origin='lower')
    plt.title("max_a Q(s,a)")
    plt.xlabel("position")
    plt.xticks(range(0,res+1,int(res/4)), numpy.around(numpy.linspace(low[0],high[0],num=5), decimals=2))
    plt.ylabel("velocity")
    plt.yticks(range(0,res+1,int(res/4)), numpy.around(numpy.linspace(low[1],high[1],num=5), decimals=3))
    plt.colorbar(orientation='vertical')

    fig.add_subplot(212)

    cmap = colors.ListedColormap(['blue', 'red', 'green'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    plot2 = plt.imshow(numpy.random.rand(res,res), origin='lower', cmap=cmap, norm=norm)
    plt.title("greedy policy")
    plt.xlabel("position")
    plt.xticks(range(0,res+1,int(res/4)),numpy.around(numpy.linspace(low[0],high[0],num=5), decimals=2))
    plt.ylabel("velocity")
    plt.yticks(range(0,res+1,int(res/4)),numpy.around(numpy.linspace(low[1],high[1],num=5), decimals=3))
    cbar = plt.colorbar(plot2, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['left', 'none', 'right'])
    plt.tight_layout()

    return plot1, plot2


# -------------------- BRAIN ---------------------------
class Brain:
    """Implementation of the neural network"""

    def __init__(self, state_count, action_count):
        self.state_count = state_count
        self.action_count = action_count

        self.model = self._create_model()
        self.model_ = self._create_model()

        if LOAD_BRAIN:
            self.model.load_weights(FILENAME_LOAD_BRAIN)
            self.model_.load_weights(FILENAME_LOAD_BRAIN)

    def _create_model(self):
        model = Sequential()

        model.add(Dense(units=128, input_dim=self.state_count, activation='relu', name='dense_1'))
        model.add(Dense(units=128, activation='relu', name='dense_2'))
        model.add(Dense(units=self.action_count, activation='linear', name='dense_out'))

        opt = RMSprop(lr=LEARNING_RATE)
        model.compile(loss='mse', optimizer=opt)

        return model

    def train(self, x, y, epoch=1, verbose=0):
        self.model.fit(x, y, batch_size=BATCH_SIZE, epochs=epoch, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predict_single(self, s, target=False):
        return self.predict(s.reshape(1, self.state_count), target=target).flatten()

    def update_target_model(self):
        self.model_.set_weights(self.model.get_weights())


# -------------------- MEMORY --------------------------
class Memory:
    """Stores previous experiences of the agent"""

    def __init__(self, capacity):
        self.samples = []
        self.capacity = capacity
        if LOAD_MEMORY:
            self.samples = self.load(FILENAME_LOAD_MEMORY)

    def add(self, sample):
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

    def is_full(self, random=False):
        if random:
            return len(self.samples) >= MEMORY_CAPACITY_RANDOM
        else:
            return len(self.samples) >= self.capacity

    def load(self, filename):
        with open(filename, 'rb') as fp:
            return pickle.load(fp)

    def save(self, filename):
        with open(filename, 'wb') as fp:
            pickle.dump(self.samples, fp)


# -------------------- AGENT ---------------------------
class Agent:
    """Interacts with the environment and uses the feedback to train its neural network"""

    def __init__(self, state_count, action_count, observation_space):
        self.state_count = state_count
        self.action_count = action_count
        self.observation_space = observation_space

        self.memory = Memory(MEMORY_CAPACITY)
        self.steps = 0
        self.epsilon = MAX_EPSILON

        self.brain = Brain(state_count, action_count)
        self.plot1, self.plot2 = initialize_plot(observation_space)

    def act(self, s):
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)
        if random.random() < self.epsilon:
            return random.randint(0, self.action_count - 1)
        else:
            return numpy.argmax(self.brain.predict_single(s))

    def observe(self, sample):
        self.memory.add(sample)

    def replay(self):
        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.update_target_model()

        batch = self.memory.sample(BATCH_SIZE)
        batch_len = len(batch)

        no_state = numpy.zeros(self.state_count)

        states = numpy.array([o[0] for o in batch])
        states_ = numpy.array([(no_state if o[3] is None else o[3]) for o in batch])

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_, target=True)

        x = numpy.zeros((batch_len, self.state_count))
        y = numpy.zeros((batch_len, self.action_count))

        for i in range(batch_len):
            o = batch[i]
            s = o[0]
            a = o[1]
            r = o[2]
            s_ = o[3]

            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * numpy.amax(p_[i])

            x[i] = s
            y[i] = t

        self.brain.train(x, y)

    def display_brain(self, res=RES):
        """Plot predictions of the neural network on the discretized state space for max_a Q(s,a) and for argmax_a Q(s,a)"""
        if self.steps % UPDATE_PLOT_FREQUENCY == 0:
            map_v, map_a = map_brain(self.brain, self.observation_space, res)

            self.plot1.set_data(map_v.T)
            self.plot1.autoscale()
            self.plot2.set_data(map_a.T)
            plt.draw()
            plt.pause(1e-17)


class RandomAgent:
    """Interacts with the environment in a random fashion to generate memory samples"""

    def __init__(self, action_count):
        self.action_count = action_count
        self.memory = Memory(MEMORY_CAPACITY)  # Full capacity since we later copy the memory to the agent
        self.steps = 0
        self.successes = 0

    def act(self, s):
        return random.randint(0, self.action_count - 1)

    def observe(self, sample):
        self.memory.add(sample)

    def replay(self):
        pass


# -------------------- ENVIRONMENT ---------------------
class Environment:
    """Controls the OpenAI Gym environment and interacts with the agent"""

    def __init__(self, gym_env):
        self.gym_env = gym_env
        self.episodes = 0
        self.last_scores = []  # For running mean reward

    def run(self, agent):
        """Run a single episode with deep Q-learning agent"""
        s = self.gym_env.reset()
        r_acc = 0

        while True:
            if RENDER:
                self.gym_env.render()

            a = agent.act(s)

            s_, r, done, _ = self.gym_env.step(a)

            if done:
                s_ = None

            if OBSERVE:
                agent.observe((s, a, r, s_))
            if TRAIN:
                agent.replay()
            if PLOT:
                agent.display_brain()

            s = s_
            agent.steps += 1
            r_acc += r

            if done:
                break

        self.last_scores.append(r_acc)
        if len(self.last_scores) > 100:
            self.last_scores.pop(0)
        mean = sum(self.last_scores) / len(self.last_scores)

        memory_len = len(agent.memory.samples)

        if self.episodes < 100:
            print('Reward: %d, steps: %d, incomplete mean: %d, memory: %d'
                % (r_acc, agent.steps, mean, memory_len))
        else:
            print('Reward: %d, steps: %d, mean: %d, memory: %d'
                    % (r_acc, agent.steps, mean, memory_len))

        self.episodes += 1

    def run_random(self, random_agent):
        """Run a single episode with randomly-acting agent"""
        s = self.gym_env.reset()
        a = random_agent.act(s)
        r_acc = 0

        while True:
            if random_agent.steps % 30 == 0:  # Sweet spot for random agent success
                a = random_agent.act(s)

            s_, r, done, _ = self.gym_env.step(a)
            

            if done:
                s_ = None

            random_agent.observe((s, a, r, s_))

            s = s_
            random_agent.steps += 1
            r_acc += r

            if done:
                break

        if r_acc > -200:
            random_agent.successes += 1


# -------------------- MAIN ----------------------------
def main():
    env = Environment(gym.make(GYM_ENV))

    state_count = env.gym_env.observation_space.shape[0]
    action_count = env.gym_env.action_space.n
    observation_space = env.gym_env.observation_space

    agent = Agent(state_count, action_count, observation_space)
    random_agent = RandomAgent(action_count)

    try:
        if MEMORY_CAPACITY_RANDOM > 0:
            print("Initialize memory with samples from random agent...")
            while not random_agent.memory.is_full(random=True):
                env.run_random(random_agent)
            print("Initilization completed, %d successful episodes." % random_agent.successes)

        agent.memory = random_agent.memory
        random_agent = None

        while True:
            env.run(agent)
    finally:
        if SAVE_BRAIN:
            agent.brain.model.save(FILENAME_SAVE_BRAIN)
        if SAVE_MEMORY:
            agent.memory.save(FILENAME_SAVE_MEMORY)

if __name__ == "__main__":
    main()