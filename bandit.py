# Simulation of the multi-armed bandit problem

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import itertools
import collections
import sympy.stats



# Agent that chooses actions according to a Gibbs/Boltzmann distribution
class SoftmaxAgent:
	def __init__(self, temperature):
		self.temperature = temperature

	def __str__(self):
		return 'Softmax agent ({:.2f})'.format(self.temperature)

	def softmax(vector):
		factors = np.exp(vector)
		total = np.sum(factors)
		probabilities = factors / total
		return probabilities

	def choose(self, actions):
		return np.random.choice(actions, p=softmax(estimates))



# Agent that chooses actions according to an epsilon-greedy strategy
# Chooses a random action with a small probability epsilon to escape local minima
class EpsilonGreedyAgent:
	def __init__(self, epsilon):
		self.epsilon = epsilon

	def __str__(self):
		return 'Epsilon-greedy agent ({:.2f})'.format(self.epsilon)

	def choose(self, actions):
		if np.random.uniform() < epsilon:
			pass # Perform random strategy
		else:
			pass # Perform greedy strategy



# (Cheating) agent that chooses the action with the highest expected reward
class OptimalAgent:
	def __init__(self, rewards):
		# The real expected rewards are 'sneaked in'
		self.rewards = rewards

	def __str__(self):
		return 'Optimal agent'

	def choose(self, actions):
		return max(actions, key = lambda action: self.rewards[action])

	def receive(self, reward):
		pass



# Agent that chooses randomly between actions
class RandomAgent:
	def __str__(self):
		return 'Random agent'

	def choose(self, actions):
		return np.random.choice(actions)

	def receive(self, reward):
		pass



# Agent that chooses the action with the highest point estimate of the expected reward
# This agent is very sensitive to the first samples it obtains for estimates and can get stuck in a suboptimal strategy
class GreedyAgent:
	def __str__(self):
		return 'Greedy agent'

	def __init__(self):
		self.totals = {} # Total reward received for each action
		self.counts = {} # Number of times each action has been performed

	def choose(self, actions):
		# Add pseudocounts, perform additive smoothing (uniform prior)
		for action in actions:
			if action not in self.totals:
				self.totals[action] = 0
			if action not in self.counts:
				self.counts[action] = 1

		# Find estimates for the expected reward of each action
		estimates = {action: self.totals[action] / self.counts[action] for action in actions}
		action = max(actions, key = lambda action: estimates[action])

		self.action = action # Remember the action that was performed when receiving the reward
		return action

	def receive(self, reward):
		self.counts[self.action] += 1
		self.totals[self.action] += reward



# Agent that chooses an action according to the probability that it maximizes the expected reward
# This one assumes rewards are sampled from a Bernoulli distribution with a beta distribution as a prior
class ThompsonAgent:
	def __str__(self):
		return 'Thompson agent'

	def __init__(self):
		self.successes = {}
		self.failures = {}

	def choose(self, actions):
		# Add pseudocounts, perform additive smoothing (uniform prior)
		for action in actions:
			if action not in self.successes:
				self.successes[action] = 1
			if action not in self.failures:
				self.failures[action] = 1

		estimates = {action: scipy.stats.beta.rvs(self.successes[action] + 1, self.failures[action] + 1) for action in actions}
		action = max(actions, key = lambda action: estimates[action])

		self.action = action # Remember the action that was performed when receiving the reward
		return action
		
	def receive(self, reward):
		# reward = scipy.stats.bernoulli(reward).rvs() # Extension to distributions with support in [0, 1]
		if reward == 0:
			self.failures[self.action] += 1
		elif reward == 1:
			self.successes[self.action] += 1



class GaussianThompsonAgent:
	def __str__(self):
		return 'Gaussian Thompson agent'

	def __init__(self):
		self.values = collections.defaultdict(lambda: [])

	def choose(self, actions):
		for action in actions:
			if len(self.values[action]) == 0:
				self.action = action
				return action

		samples = {}
		for action in actions:
			params = scipy.stats.norm.fit(self.values[action])
			distribution = scipy.stats.norm(*params)
			# Sample the mean of this distribution
			samples[action] = np.mean(distribution.rvs(size=20))

		action = max(actions, key = lambda action: samples[action])
		self.action = action
		return action

	def receive(self, reward):
		self.values[self.action].append(reward)



# List of all possible actions
actions = [scipy.stats.bernoulli(scipy.stats.uniform().rvs()) for arm in range(10)]
actions = [scipy.stats.norm(scipy.stats.norm.rvs(), np.exp(scipy.stats.norm.rvs())) for arm in range(10)]
actions = [scipy.stats.poisson(np.exp(scipy.stats.norm.rvs())) for arm in range(10)]
actions = [scipy.stats.lognorm(np.exp(scipy.stats.norm.rvs())) for arm in range(10)]
actions = [scipy.stats.gamma(np.exp(scipy.stats.norm.rvs())) for arm in range(10)]

# Expected reward for each action
rewards = {action: action.expect() for action in actions}
best_reward = max(rewards.values())
print('Expected rewards: ' + ', '.join('{:.2f}'.format(value) for value in rewards.values()))
print('Best expected reward: {:.2f}'.format(best_reward))

# List of agents
agents = [OptimalAgent(rewards), RandomAgent(), GreedyAgent(), ThompsonAgent(), GaussianThompsonAgent()]

# History of rewards received by each agent
rewards = {agent: [] for agent in agents}

# Simulation
rounds = range(1000)
for t in rounds:
	for agent in agents:
		# Find action chosen by agent
		action = agent.choose(actions)

		# Find reward returned by chosen action (sample the reward distribution for that action)
		reward = action.rvs()

		# Reward agent for chosen action
		agent.receive(reward)

		# Add reward to history of rewards for this agent
		rewards[agent].append(reward)

# Find the expected total reward under the best expected return
best_accumulated_rewards = [t * best_reward for t in rounds]

# Find average regret per round (difference between best expected total reward and actual total reward, divided by number of rounds)
for agent in agents:
	accumulated_rewards = list(itertools.accumulate(rewards[agent]))
	accumulated_regrets = [best_accumulated_rewards[t] - accumulated_rewards[t] for t in rounds]
	average_regrets = [accumulated_regrets[t] / (t + 1) for t in rounds]
	plt.plot(average_regrets, label = str(agent))

plt.title('Multi-armed bandit problem with {} arms'.format(len(actions)))
plt.xlabel('Rounds')
plt.ylabel('Average regret per round')
plt.legend(loc='best')
plt.show()