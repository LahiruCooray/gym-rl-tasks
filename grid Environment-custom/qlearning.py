# Import necessary libraries
import numpy as np
from PIL import Image
import cv2
from matplotlib import style
import matplotlib.pyplot as plt
import pickle
import time

# Set matplotlib plot style
style.use('ggplot')

# === Environment Parameters ===
SIZE = 10                      # Size of the 2D grid environment
HM_EPISODES = 25000           # Number of episodes for training
MOVE_PENALTY = 1              # Penalty for each move
ENEMY_PENALTY = 300           # Penalty if player meets enemy
FOOD_REWARD = 25              # Reward for reaching the food

# === Exploration Parameters ===
epsilon = 0                   # Epsilon value for epsilon-greedy exploration
EPS_DECAY = 0.9998            # Decay rate of epsilon
SHOW_EVERY = 1                # Show environment every n episodes

# === Q-Table & Learning Parameters ===
start_q_table = "q_table-1748171555.pickle"  # Pretrained Q-table
LEARNING_RATE = 0.1
DISCOUNT = 0.95               # Future reward discount factor

# === Object Identifiers for Rendering ===
PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3

# === Color Dictionary for Visualization ===
d = {
    PLAYER_N: (255, 0, 0),     # Blue for Player
    FOOD_N: (0, 255, 0),       # Green for Food
    ENEMY_N: (0, 0, 255)       # Red for Enemy
}

# === Blob Class: Defines player, enemy, and food ===
class Blob:
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)
    
    def __str__(self):
        return f"({self.x}, {self.y})"
    
    # Subtraction operator to get relative position
    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)
    
    # Take an action (encoded as integer)
    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)
    
    # Move blob with optional fixed direction or random
    def move(self, x=False, y=False):
        self.x += x if x is not False else np.random.randint(-1, 2)
        self.y += y if y is not False else np.random.randint(-1, 2)

        # Keep within grid boundaries
        self.x = max(0, min(self.x, SIZE - 1))
        self.y = max(0, min(self.y, SIZE - 1))

# === Initialize or Load Q-table ===
if start_q_table is None:
    q_table = {}
    for x1 in range(-SIZE + 1, SIZE):
        for y1 in range(-SIZE + 1, SIZE):
            for x2 in range(-SIZE + 1, SIZE):
                for y2 in range(-SIZE + 1, SIZE):
                    q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for _ in range(4)]
else:
    with open(start_q_table, 'rb') as f:
        q_table = pickle.load(f)

# === Store rewards per episode for plotting ===
episode_rewards = []

# === Main training loop ===
for episode in range(HM_EPISODES):
    player = Blob()
    food = Blob()
    enemy = Blob()

    # Display environment every SHOW_EVERY episodes
    show = episode % SHOW_EVERY == 0
    if show:
        print(f"on # {episode}, epsilon: {epsilon}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
    
    episode_reward = 0

    for i in range(200):  # Max 200 steps per episode
        obs = (player - food, player - enemy)

        # Epsilon-greedy action selection
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)

        player.action(action)

        # Reward assignment
        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
            food = Blob()
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
            enemy = Blob()
        else:
            reward = -MOVE_PENALTY

        # Q-learning update rule
        new_obs = (player - food, player - enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        elif reward == -ENEMY_PENALTY:
            new_q = -ENEMY_PENALTY
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        
        q_table[obs][action] = new_q

        # Visualization
        if show:
            env_img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            env_img[food.x, food.y] = d[FOOD_N]
            env_img[player.x, player.y] = d[PLAYER_N]
            env_img[enemy.x, enemy.y] = d[ENEMY_N]

            img = Image.fromarray(env_img, 'RGB')
            img = img.resize((300, 300), Image.NEAREST)
            cv2.imshow("Environment", np.array(img))

            delay = 500 if reward in [FOOD_REWARD, -ENEMY_PENALTY] else 1
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

        episode_reward += reward

        # Break episode if terminal condition met
        if reward in [FOOD_REWARD, -ENEMY_PENALTY]:
            break
    
    # End of episode updates
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY  # Decay epsilon

# === Plotting average rewards ===
moving_average = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
plt.plot([i for i in range(len(moving_average))], moving_average)
plt.ylabel(f"Reward {SHOW_EVERY} episode moving average")
plt.xlabel("Episode #")
plt.show()

# === Save trained Q-table ===
print("Saving Q-table...")
with open(f"q_table-{int(time.time())}.pickle", 'wb') as f:
    pickle.dump(q_table, f)
print("Q-table saved.")
