import gym
import numpy as np
import random

# Create the FrozenLake environment
env = gym.make('FrozenLake-v1', is_slippery=False)

# Initialize the Q-table
num_states = env.observation_space.n
num_actions = env.action_space.n
Q_table = np.zeros((num_states, num_actions))

# Parameters
total_episodes = 1000
learning_rate = 0.8
max_steps = 99
gamma = 0.95
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.01

# The Q-learning algorithm
for episode in range(total_episodes):
    state = env.reset()
    done = False
    for step in range(max_steps):
        # Choose an action in the current world state (s)
        # First we randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)

        # If this number > epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(Q_table[state,:])
        # Else doing a random choice --> exploration
        else:
            action = env.action_space.sample()

        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env.step(action)

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        Q_table[state, action] = Q_table[state, action] + learning_rate * (reward + gamma * np.max(Q_table[new_state, :]) - Q_table[state, action])

        # Our new state is state
        state = new_state

        # If done: finish episode
        if done:
            break

    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

# Print the Q-table
print("Q-table:")
print(Q_table)

