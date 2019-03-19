"""Deep Q-network implementation for the OpenAI Gym MountainCar-v0 environment.

We use a feedforward neural network with two fully-connected hidden layers
that consist of 128 neurons each. The model is trained via experience replay
by sampling randomly from the memory. The target Q function is given by
a second neural network with identical properties, whose weights are regularly
updated by the main model. The mean reward after training is around -100.

To speed up the learning, we initially fill the memory by taking random
actions. This gives the agent information about the position of the goal.
We also normalize the state variables to the interval [-1,1].

Due to the two-dimensional state space (position and velocity), we can
visualize the neural network while it learns. We do this by discretizing
the state space and plotting the prediction of our model for the value
function max_a Q(s,a) as well as for the greedy policy argmax_a Q(s,a).
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
LEARNING_RATE = 0.00025
UPDATE_TARGET_FREQUENCY = 1000  # Step rate for copying weights from the main model to the target model

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
def map_brain(brain, res=RES):
    """Calculates the preciditions of the neural network for the discretized state space.

    Args:
        brain: Implementation of the neural network.
        res: Resolution per variable for state space discretization.

    Returns:
        Numpy arrays for max_a Q(s,a) and for argmax_a Q(s,a) on the discretized state space.
    """
    s = numpy.zeros((res * res, 2))
    i = 0

    # we have normalized the states in the environment
    low = [-1,-1]
    high = [1,1]

    for i1 in range(res):
        for i2 in range(res):
            s[i] = numpy.array([low[0] + (high[0]-low[0]) * i1 / (res - 1),
                                low[1] + (high[1]-low[1]) * i2 / (res - 1)])
            i += 1

    map_v = numpy.amax(brain.predict(s), axis=1).reshape((res, res))
    map_a = numpy.argmax(brain.predict(s), axis=1).reshape((res, res))

    return map_v, map_a

def initialize_plot(observation_space, res=RES):
    """Creates two subplots for later visualization of the neural network predictions.

    Args:
        observation_space: Observation space of the OpenAI Gym environment.
        res: Resolution per variable for state space discretization.
    Returns:
        Two imshow objects for max_a Q(s,a) and for argmax_a Q(s,a).
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
    plt.title("argmax_a Q(s,a)")
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
    """Implementation of the main and the target neural network.
    
    Args:
        state_count: Number of state variables.
        action_count: Number of possible actions.
    """

    def __init__(self, state_count, action_count):
        """Initializes the main and the target neural network."""
        self._state_count = state_count
        self._action_count = action_count

        self._model = self._create_model()
        self._target_model = self._create_model()

        if LOAD_BRAIN:
            self._model.load_weights(FILENAME_LOAD_BRAIN)
            self._target_model.load_weights(FILENAME_LOAD_BRAIN)

    def _create_model(self):
        """Creates a feedforward neural network with Keras."""
        model = Sequential()

        model.add(Dense(units=128, input_dim=self._state_count, activation='relu', name='dense_1'))
        model.add(Dense(units=128, activation='relu', name='dense_2'))
        model.add(Dense(units=self._action_count, activation='linear', name='dense_out'))

        opt = RMSprop(lr=LEARNING_RATE)
        model.compile(loss='mse', optimizer=opt)

        return model

    def train(self, x, y, epoch=1, verbose=0):
        """Trains the neural network.

        Args:
            x: Array of states that are used as input variables.
            y: Array of the target function Q(s,a) for the states s and
                all possible actions a.
            epoch: Number of epochs to use for training.
            verbose: Set verbosity of the fitting function.
        """
        self._model.fit(x, y, batch_size=BATCH_SIZE, epochs=epoch, verbose=verbose)

    def predict(self, s, target=False):
        """Returns a prediction of Q(s,a) for a given set of states s and all possible actions a.

        Args:
            s: Array of states.
            target: Whether to use the target network for the prediction.
        """
        if target:
            return self._target_model.predict(s)
        else:
            return self._model.predict(s)

    def predict_single(self, s, target=False):
        """Returns a prediction of Q(s,a) for a single state s and all possible actions a.

        Args:
            s: Single state.
            target: Whether to use the target network for the prediction.
        """
        return self.predict(s.reshape(1, self._state_count), target=target).flatten()

    def update_target_model(self):
        """Copies the weights of the main neural network to the target network."""
        self._target_model.set_weights(self._model.get_weights())

    def save(self, filename):
        """Saves the weights of the neural network to a file."""
        self._model.save(filename)


# -------------------- MEMORY --------------------------
class Memory:
    """Stores previous experiences of the agent.
    
    Args:
        capacity: Maximum number of experiences that can be saved.
    """

    def __init__(self, capacity):
        """Initializes the memory."""
        self._samples = []
        self._capacity = capacity
        if LOAD_MEMORY:
            self._samples = self.load(FILENAME_LOAD_MEMORY)

    def add(self, sample):
        """Adds an experience to the memory.

        Args:
            sample: Single experience in the format (old_state, action, reward, new_state).
        """
        self._samples.append(sample)
        if len(self._samples) > self._capacity:
            self._samples.pop(0)

    def sample(self, n):
        """Randomly sample experiences from the memory.

        Args:
            n: Number of experiences to sample.
        
        Returns:
            A list of experiences, which in turn are lists in the format (old_state, action, reward, new_state).
        """
        n = min(n, len(self._samples))
        return random.sample(self._samples, n)

    def is_full(self, random=False):
        """Checks whether the memory is full.

        Args:
            random: Whether to consider the maximum memory size for the random agent.
        """
        if random:
            return len(self._samples) >= MEMORY_CAPACITY_RANDOM
        else:
            return len(self._samples) >= self._capacity
    
    def size(self):
        """Returns the number of experiences in the memory."""
        return len(self._samples)

    def load(self, filename):
        """Loads the memory from a file.

        Returns:
            List of experiences.
        """
        with open(filename, 'rb') as fp:
            return pickle.load(fp)

    def save(self, filename):
        """Saves the memory to a file."""
        with open(filename, 'wb') as fp:
            pickle.dump(self._samples, fp)


# -------------------- AGENT ---------------------------
class Agent:
    """Interacts with the environment and uses the feedback to train its neural network.
    
    Args:
        state_count: Number of state variables.
        action_count: Number of possible actions.
        observation_space: Observation space of the OpenAI Gym environment.
    """

    def __init__(self, state_count, action_count, observation_space):
        """Initializes the agent."""
        self.state_count = state_count
        self.action_count = action_count

        self.memory = Memory(MEMORY_CAPACITY)
        self.steps = 0
        self.epsilon = MAX_EPSILON

        self.brain = Brain(state_count, action_count)
        self.plot1, self.plot2 = initialize_plot(observation_space)

    def act(self, s):
        """Returns an action based on the current state and the epsilon-greedy policy.

        Args:
            s: Current state.
        """
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)
        if random.random() < self.epsilon:
            return random.randint(0, self.action_count - 1)
        else:
            return numpy.argmax(self.brain.predict_single(s))

    def observe(self, sample):
        """Adds an experience to the memory."""
        self.memory.add(sample)

    def replay(self):
        """Trains the neural network with a mini-batch that is randomly sampled from the memory."""
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
        """Plots predictions of the neural network on the discretized state space
        for the value function max_a Q(s,a) and the greedy policy argmax_a Q(s,a).
        
        Args:
            res: Resolution per variable for state space discretization.
        """
        if self.steps % UPDATE_PLOT_FREQUENCY == 0:
            map_v, map_a = map_brain(self.brain, res)

            self.plot1.set_data(map_v.T)
            self.plot1.autoscale()
            self.plot2.set_data(map_a.T)
            plt.draw()
            plt.pause(1e-17)


class RandomAgent:
    """Interacts with the environment in a random fashion to generate memory samples.
    
    Args:
        action_count: Number of possible actions.
    """

    def __init__(self, action_count):
        """Initializes the random agent."""
        self.action_count = action_count
        self.memory = Memory(MEMORY_CAPACITY)  # Full capacity since we later copy the memory to the agent
        self.steps = 0
        self.successes = 0

    def act(self, s):
        """Returns a random action.

        Args:
            s: Current state.
        """
        return random.randint(0, self.action_count - 1)

    def observe(self, sample):
        """Adds an experience to the memory."""
        self.memory.add(sample)


# -------------------- ENVIRONMENT ---------------------
class Environment:
    """Controls the OpenAI Gym environment and interacts with the agent.
    
    Args:
        gym_env: OpenAI Gym environment.
    """

    def __init__(self, gym_env):
        """Initializes the environment."""
        self.gym_env = gym_env
        self.episodes = 0
        self.last_scores = []  # For running mean reward

        low = gym_env.observation_space.low
        high = gym_env.observation_space.high
        self.mean = (high + low) / 2
        self.spread = abs(high - low) / 2

    def _normalize(self, s):
        """Normalizes the state variables to the interval [-1,1]."""
        return (s - self.mean) / self.spread

    def run(self, agent):
        """Runs a single episode with deep Q-learning agent.
        
        Args:
            agent: Instance of Agent.
        """
        s = self.gym_env.reset()
        s = self._normalize(s)
        r_acc = 0

        while True:
            if RENDER:
                self.gym_env.render()

            a = agent.act(s)

            s_, r, done, _ = self.gym_env.step(a)
            s_ = self._normalize(s_)

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

        # Calculate running mean reward
        self.last_scores.append(r_acc)
        if len(self.last_scores) > 100:
            self.last_scores.pop(0)
        running_mean = sum(self.last_scores) / len(self.last_scores)

        if self.episodes < 100:
            print('Reward: %d, steps: %d, incomplete mean: %d, memory: %d'
                % (r_acc, agent.steps, running_mean, agent.memory.size()))
        else:
            print('Reward: %d, steps: %d, mean: %d, memory: %d'
                    % (r_acc, agent.steps, running_mean, agent.memory.size()))

        self.episodes += 1

    def run_random(self, random_agent):
        """Runs a single episode with randomly-acting agent.
        
        Args:
            random_agent: Instance of RandomAgent."""
        s = self.gym_env.reset()
        s = self._normalize(s)
        a = random_agent.act(s)
        r_acc = 0

        while True:
            if random_agent.steps % 30 == 0:  # Sweet spot for random agent success
                a = random_agent.act(s)

            s_, r, done, _ = self.gym_env.step(a)
            s_ = self._normalize(s_)

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
    """Initializes and runs the OpenAI Gym environment."""
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
            agent.brain.save(FILENAME_SAVE_BRAIN)
        if SAVE_MEMORY:
            agent.memory.save(FILENAME_SAVE_MEMORY)

if __name__ == "__main__":
    main()
