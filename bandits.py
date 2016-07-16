import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import itertools

# Add softmax and epsilon-greedy agents

# A (cheating) agent that chooses the action with the highest expected reward
class OptimalAgent:
	def __str__(self):
		return 'Optimal agent'
	def __init__(self, returns):
		# The real expected returns are 'sneaked in'
		self.returns = returns
	def choose(self, actions):
		return max(actions, key = lambda action: self.returns[action])
	def receive(self, reward):
		pass

# An agent that chooses randomly between actions
class RandomAgent:
	def __str__(self):
		return 'Random agent'
	def choose(self, actions):
		return np.random.choice(actions)
	def receive(self, reward):
		pass

# An agent that chooses the action with the highest point estimate of the expected reward
# This agent is very sensitive to the first few samples it uses for estimates
class GreedyAgentOld:
	def __str__(self):
		return 'Greedy agent'
	def __init__(self):
		self.action_history = []
		self.reward_history = []
	def choose(self, actions):
		# Point estimates for the expected reward of each action
		totals = {action: 0 for action in actions}
		counts = {action: 0 for action in actions}
		for action, reward in zip(self.action_history, self.reward_history):
			totals[action] += reward
			counts[action] += 1
		estimates = {action: totals[action] / (counts[action] + 1) for action in actions}
		action = max(actions, key = lambda action: estimates[action])
		self.action_history.append(action)
		return action
	def receive(self, reward):
		self.reward_history.append(reward)

class GreedyAgent:
	def __str__(self):
		return 'Greedy agent'
	def __init__(self):
		self.totals = {} # Total reward that has been received for each action
		self.counts = {} # Number of times each action has been performed
	def choose(self, actions):
		# Point estimates for the expected reward of each action
		for action in actions:
			if action not in self.totals:
				self.totals[action] = 0
			if action not in self.counts:
				self.counts[action] = 1

		estimates = {action: self.totals[action] / self.counts[action] for action in actions}
		action = max(actions, key = lambda action: estimates[action])
		self.counts[action] += 1
		self.last_action = action # This is the last action that was performed
		return action
	def receive(self, reward):
		self.totals[self.last_action] += reward

# An agent that chooses an action according to the probability that it maximizes the expected reward
class ThompsonAgent:
	def __str__(self):
		return 'Thompson agent'
	def __init__(self):
		self.action_history = []
		self.reward_history = []
	def choose(self, actions):
		successes = {action: 0 for action in actions}
		failures = {action: 0 for action in actions}
		for action, reward in zip(self.action_history, self.reward_history):
			if reward == 1:
				successes[action] += 1
			elif reward == 0:
				failures[action] += 1
		estimates = {action: scipy.stats.beta.rvs(successes[action] + 1, failures[action] + 1) for action in actions}
		action = max(actions, key = lambda action: estimates[action])
		self.action_history.append(action)
		return action
	def receive(self, reward):
		self.reward_history.append(reward)

# List of all possible actions
number_of_actions = 10
actions = [scipy.stats.bernoulli(scipy.stats.uniform().rvs()) for n in range(number_of_actions)]

# Expected returns for each action
returns = {action: action.expect() for action in actions}
best_return = max(returns.values())
print('Expected returns: ' + ', '.join('{:.2f}'.format(value) for value in returns.values()))
print('Best expected return: ' + '{:.2f}'.format(best_return))

# List of agents
agents = [OptimalAgent(returns), RandomAgent(), GreedyAgent(), ThompsonAgent()]

# History of rewards received by each agent
rewards = {}
for agent in agents:
	rewards[agent] = []

# Simulation
number_of_rounds = 1000
for rounds in range(number_of_rounds):
	for agent in agents:

		# Find action chosen by agent
		choice = agent.choose(actions)

		# Find reward returned by chosen action
		reward = choice.rvs()

		# Reward agent for chosen action
		agent.receive(reward)

		# Add reward to history of rewards for this agent
		rewards[agent].append(reward)

optimal_cumulative_rewards = [best_return * round_number for round_number in range(number_of_rounds)]

plt.title('Multi-armed bandit problem with ' + str(number_of_actions) + ' arms')
for agent in agents:
	cumulative_rewards = list(itertools.accumulate(rewards[agent]))
	cumulative_regret = [optimal_cumulative_rewards[round_number] - cumulative_rewards[round_number] for round_number in range(number_of_rounds)]
	average_regret = [cumulative_regret[round_number] / (round_number + 1) for round_number in range(number_of_rounds)]
	plt.plot(cumulative_regret, label = str(agent))

plt.xlabel('Rounds')
plt.ylabel('Average regret per round')
# plt.yscale('log')
plt.legend()
plt.show()