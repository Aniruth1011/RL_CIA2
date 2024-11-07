import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from queue import PriorityQueue
from tqdm import tqdm

class GridWorld:
    def __init__(self, size, start, goal, obstacle_prob=0.3):
        self.size = size
        self.start = start
        self.goal = goal
        self.grid = np.zeros((size, size))
        self.grid[goal] = 2
        print("Initializing grid with obstacles...")
        for i in range(size):
            for j in range(size):
                if random.random() < obstacle_prob and (i, j) != start and (i, j) != goal:
                    self.grid[i, j] = -1
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    def get_neighbors(self, state):
        neighbors = []
        for action in self.actions:
            new_state = (state[0] + action[0], state[1] + action[1])
            if 0 <= new_state[0] < self.size and 0 <= new_state[1] < self.size and self.grid[new_state] != -1:
                neighbors.append(new_state)
        return neighbors

class MDPAgent:
    def __init__(self, gridworld):
        self.gridworld = gridworld
        self.values = np.zeros((gridworld.size, gridworld.size))
        self.policy = np.zeros((gridworld.size, gridworld.size, len(gridworld.actions)))
    def policy_evaluation(self, gamma=0.9, threshold=1e-3):
        print("Starting policy evaluation...")
        while True:
            delta = 0
            for i in tqdm(range(self.gridworld.size), desc="Policy Evaluation Rows"):
                for j in range(self.gridworld.size):
                    if (i, j) == self.gridworld.goal:
                        continue
                    v = self.values[i, j]
                    self.values[i, j] = max([sum([0.25 * (reward + gamma * self.values[ni, nj]) for (ni, nj) in self.gridworld.get_neighbors((i, j))]) for reward in [1]])
                    delta = max(delta, abs(v - self.values[i, j]))
            if delta < threshold:
                break
        print("Policy evaluation completed.")
    def extract_policy(self, gamma=0.9):
        print("Extracting optimal policy...")
        for i in tqdm(range(self.gridworld.size), desc="Extracting Policy Rows"):
            for j in range(self.gridworld.size):
                if (i, j) == self.gridworld.goal:
                    continue
                q_values = []
                for action in self.gridworld.actions:
                    next_state = (i + action[0], j + action[1])
                    if 0 <= next_state[0] < self.gridworld.size and 0 <= next_state[1] < self.gridworld.size and self.gridworld.grid[next_state] != -1:
                        q_values.append(self.values[next_state])
                if q_values: 
                    self.policy[i, j] = np.eye(len(self.gridworld.actions))[np.argmax(q_values)]
                else:
                    self.policy[i, j] = np.zeros(len(self.gridworld.actions))
        print("Policy extraction completed.")

class QLearningAgent:
    def __init__(self, gridworld, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.gridworld = gridworld
        self.q_table = defaultdict(lambda: np.zeros(len(gridworld.actions)))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(range(len(self.gridworld.actions)))
        return np.argmax(self.q_table[state])
    def learn(self, episodes=1000):
        print("Starting Q-learning...")
        for episode in tqdm(range(episodes), desc="Q-learning Episodes"):
            state = self.gridworld.start
            while state != self.gridworld.goal:
                action = self.choose_action(state)
                next_state = (state[0] + self.gridworld.actions[action][0], state[1] + self.gridworld.actions[action][1])
                if 0 <= next_state[0] < self.gridworld.size and 0 <= next_state[1] < self.gridworld.size and self.gridworld.grid[next_state] != -1:
                    reward = 1 if next_state == self.gridworld.goal else -0.1
                    best_next_action = np.argmax(self.q_table[next_state])
                    td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
                    td_delta = td_target - self.q_table[state][action]
                    self.q_table[state][action] += self.alpha * td_delta
                    state = next_state
        print("Q-learning completed.")

def dijkstra(gridworld):
    print("Starting Dijkstra's algorithm...")
    size = gridworld.size
    dist = { (i, j): float('inf') for i in range(size) for j in range(size) }
    dist[gridworld.start] = 0
    prev = { (i, j): None for i in range(size) for j in range(size) }
    pq = PriorityQueue()
    pq.put((0, gridworld.start))
    while not pq.empty():
        _, current = pq.get()
        if current == gridworld.goal:
            break
        for neighbor in gridworld.get_neighbors(current):
            alt = dist[current] + 1
            if alt < dist[neighbor]:
                dist[neighbor] = alt
                prev[neighbor] = current
                pq.put((alt, neighbor))
    path = []
    step = gridworld.goal
    while step is not None:
        path.append(step)
        step = prev[step]
    path.reverse()
    print("Dijkstra's algorithm completed.")
    return path

size = 20
start = (0, 0)
goal = (size - 1, size - 1)
gridworld = GridWorld(size, start, goal)

mdp_agent = MDPAgent(gridworld)
mdp_agent.policy_evaluation()
mdp_agent.extract_policy()

q_agent = QLearningAgent(gridworld)
q_agent.learn()

path_dijkstra = dijkstra(gridworld)

print("Visualizing the path and saving as PNG...")
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(gridworld.grid, cmap='gray')
for (x, y) in path_dijkstra:
    ax.plot(y, x, 'ro', markersize=1)
ax.plot(start[1], start[0], 'go', markersize=10)
ax.plot(goal[1], goal[0], 'bo', markersize=10)
plt.savefig('optimal_path.png')
print("Path visualization saved as 'optimal_path.png'.")
