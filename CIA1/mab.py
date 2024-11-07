import numpy as np

class EpsilonGreedyBandit:
    def __init__(self, n_actions, epsilon=0.1):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.counts = np.zeros(n_actions)  # act num
        self.values = np.zeros(n_actions)  
        self.total_counts = 0 

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n_actions) #expl
        else:
            return np.argmax(self.values) #expllot
    
    def update(self, action, reward):
        self.total_counts += 1
        self.counts[action] += 1
        self.values[action] += (reward - self.values[action]) / self.counts[action]

n_actions = 10 
epsilon = 0.1  
n_steps = 10000  

bandit = EpsilonGreedyBandit(n_actions, epsilon)

true_action_values = np.random.normal(0, 1, n_actions)
true_action_values[3] += 1.5 
true_action_values[7] += 1.0  

for step in range(n_steps): 
    action = bandit.select_action()
    reward = np.random.normal(true_action_values[action], 1.0)
    bandit.update(action, reward)

print("Action counts:", bandit.counts)
print("Estimated rewards:", bandit.values)
print("True action values:", true_action_values)
