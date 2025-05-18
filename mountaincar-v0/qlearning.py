import gym
import numpy as np
import random
import os
import matplotlib.pyplot as plt

# Create environment
env = gym.make("MountainCar-v0", render_mode="rgb_array")
env.reset()

# Hyperparameters
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 500

# Exploration settings
epsilon = 0.5  # Exploration rate
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# Discretize state space
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# File paths
Q_TABLE_FILE = "q_table.npy"
EPISODE_FILE = "episode.npy"
EPSILON_FILE = "epsilon.npy"

# Load saved Q-table if exists
if os.path.exists(Q_TABLE_FILE):
    q_table = np.load(Q_TABLE_FILE)
    start_episode = int(np.load(EPISODE_FILE))
    epsilon = float(np.load(EPSILON_FILE)) if os.path.exists(EPSILON_FILE) else epsilon
    print(f"Resuming from episode {start_episode}")
else:
    q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n])) #(20,20,3)
    start_episode = 0
    print("Starting fresh...")

ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))

# Start training loop from last episode
for episode in range(start_episode, EPISODES):
    episode_reward = 0
    if episode % SHOW_EVERY == 0:
        print(f"Episode: {episode}, epsilon: {epsilon:.2f}")
        render = True
    else:
        render = False

    state, _ = env.reset()
    discrete_state = get_discrete_state(state)
    done = False

    while not done:
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[discrete_state])  # Exploit

        new_state, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        done = terminated or truncated
        new_discrete_state = get_discrete_state(new_state)

        if render:
            env.render()

        # Q-learning update
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q
        elif new_state[0] >= env.unwrapped.goal_position:
            print(f"Goal Reached! on episode: {episode}")
            q_table[discrete_state + (action,)] = 0  # Reward the goal-reaching move

        discrete_state = new_discrete_state

    # Decay epsilon
    if START_EPSILON_DECAYING <= episode < END_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    # Save progress periodically
    if episode % 10 == 0:
        np.save(f"qtables/q_table_{episode}.npy", q_table)
        np.save(Q_TABLE_FILE, q_table)
        np.save(EPISODE_FILE, episode)
        np.save(EPSILON_FILE, epsilon)
    
    ep_rewards.append(episode_reward)
    if not episode % SHOW_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY:]) / len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))
        print(f"Episode: {episode}, Avg: {average_reward:.2f}, Min: {min(ep_rewards[-SHOW_EVERY:])}, Max: {max(ep_rewards[-SHOW_EVERY:])}")

env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max')
plt.legend(loc='upper left')
plt.savefig("training_rewards_plot.png")

plt.show()
