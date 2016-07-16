import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import itertools

# Add softmax and epsilon-greedy agents



# A (cheating) agent that chooses the action with the highest expected reward
class OptimalAgent:
	def __str__(self):
		return 'Optimal agent'

	def __init__(self, rewards):
		# The real expected rewards are 'sneaked in'
		self.rewards = rewards

	def choose(self, actions):
		return max(actions, key = lambda action: self.rewards[action])

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
# This agent is very sensitive to the first samples it obtains for estimates and can get stuck in a suboptimal strategy
class GreedyAgent:
	def __str__(self):
		return 'Greedy agent'

	def __init__(self):
		self.totals = {} # Total reward received for each action
		self.counts = {} # Number of times each action has been performed

	def choose(self, actions):
		# Add pseudocounts, perform additive smoothing
		for action in actions:
			if action not in self.totals:
				self.totals[action] = 0
			if action not in self.counts:
				self.counts[action] = 1

		# Find estimates for the expected reward of each action
		estimates = {action: self.totals[action] / self.counts[action] for action in actions}
		action = max(actions, key = lambda action: estimates[action])

		self.counts[action] += 1
		self.last_action = action # Remember the action that was performed when receiving the reward
		return action

	def receive(self, reward):
		self.totals[self.last_action] += reward



# An agent that chooses an action according to the probability that it maximizes the expected reward
class ThompsonAgent:
	def __str__(self):
		return 'Thompson agent'

	def __init__(self):
		self.successes = {}
		self.failures = {}

	def choose(self, actions):
		# Add pseudocounts, perform additive smoothing
		for action in actions:
			if action not in self.successes:
				self.successes[action] = 1
			if action not in self.failures:
				self.failures[action] = 1

		estimates = {action: scipy.stats.beta.rvs(self.successes[action] + 1, self.failures[action] + 1) for action in actions}
		action = max(actions, key = lambda action: estimates[action])

		self.last_action = action
		return action
		
	def receive(self, reward):
		if reward == 0:
			self.failures[self.last_action] += 1
		elif reward == 1:
			self.successes[self.last_action] += 1



# List of all possible actions
actions = [scipy.stats.bernoulli(scipy.stats.uniform().rvs()) for arm in range(10)]

# Expected reward for each action
rewards = {action: action.expect() for action in actions}
best_reward = max(rewards.values())
print('Expected rewards: ' + ', '.join('{:.2f}'.format(value) for value in rewards.values()))
print('Best expected reward: {:.2f}'.format(best_reward))

# List of agents
agents = [OptimalAgent(rewards), RandomAgent(), GreedyAgent(), ThompsonAgent()]

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
	accumulated_regret = [best_accumulated_rewards[t] - accumulated_rewards[t] for t in rounds]
	average_regret = [accumulated_regret[t] / (t + 1) for t in rounds]
	plt.plot(average_regret, label = str(agent))

plt.title('Multi-armed bandit problem with {} arms'.format(len(actions)))
plt.xlabel('Rounds')
plt.ylabel('Average regret per round')
plt.legend()
plt.show()