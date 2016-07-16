import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import itertools

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
class GreedyAgent:
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
print('Expected returns: ' + ', '.join('{:.2f}'.format(value) for value in returns.values()))

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

plt.title('Multi-armed bandit problem with ' + str(number_of_actions) + ' arms')
for agent in agents:
	plt.plot(list(itertools.accumulate(rewards[agent])), label = str(agent))
plt.xlabel('Rounds')
plt.ylabel('Cumulative reward')
plt.legend()
plt.show()








exit()


import numpy as np

class Agent:
	def __init__(self):
		self.rewards = []

	def choose(self, actions):
		raise NotImplementedError

	def receive(self, reward):
		raise NotImplementedError





class RandomAgent(Agent):
	def choose(self, actions):
		return np.random.choice(actions)






agent = RandomAgent()

print(agent.choose(['move left', 'move right']))











exit()



import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import itertools


# Problem with greedy statregy = 
# does not take into consideration
# the confidence for each point esimate
# it judges too quickly

arms = 100

distributions = [scipy.stats.bernoulli(scipy.stats.uniform().rvs()) for arm in range(arms)]

agents = {'optimal', 'random', 'greedy', 'semi-greedy', 'thompson'}

histories = {agent: [] for agent in agents}

plt.ion()
plt.show()
# plt.yscale('log')

step = 0
skip = 100
while True:
	for agent in agents:
		if agent == 'optimal':
			choice = max(distributions, key = lambda distribution: distribution.expect())
			histories[agent].append((choice, choice.rvs()))
			if step % skip == 0: plt.plot(tuple(itertools.accumulate([event[1] for event in histories[agent]])), color = 'red')
		elif agent == 'random':
			choice = np.random.choice(distributions)
			histories[agent].append((choice, choice.rvs()))
			if step % skip == 0: plt.plot(tuple(itertools.accumulate([event[1] for event in histories[agent]])), color = 'green')
		elif agent == 'greedy':
			history = histories[agent]
			estimates = {}
			for distribution in distributions:
				successes = sum(event[1] for event in history if event[0] == distribution)
	
				total = len([event for event in history if event[0] == distribution])
				estimate = (successes + .5) / (total + 1)
				estimates[distribution] = estimate
			choice = max(distributions, key = lambda distribution: estimates[distribution])
			histories[agent].append((choice, choice.rvs()))
			if step % skip == 0: plt.plot(tuple(itertools.accumulate([event[1] for event in histories[agent]])), color = 'blue')
		elif agent == 'semi-greedy':
			exploration_probability = np.exp(-len(histories) / arms)
			if scipy.stats.uniform.rvs() < exploration_probability:
				# Perform greedy strategy
				history = histories[agent]
				estimates = {}
				for distribution in distributions:
					successes = sum(event[1] for event in history if event[0] == distribution)
		
					total = len([event for event in history if event[0] == distribution])
					estimate = (successes + .5) / (total + 1)
					estimates[distribution] = estimate
				choice = max(distributions, key = lambda distribution: estimates[distribution])
			else:
				# Select random machine
				choice = np.random.choice(distributions)
			histories[agent].append((choice, choice.rvs()))
			if step % skip == 0: plt.plot(tuple(itertools.accumulate([event[1] for event in histories[agent]])), color = 'purple')
		elif agent == 'thompson':
			history = histories[agent]
			successes = {}
			failures = {}
			for distribution in distributions:
				successes[distribution] = 0
				failures[distribution] = 0
			for event in history:
				choice, reward = event
				if reward == 1:
					successes[choice] += 1
				elif reward == 0:
					failures[choice] += 1
			mean_samples = {}
			for distribution in distributions:
				mean_samples[distribution] = scipy.stats.beta.rvs(successes[distribution] + 1, failures[distribution] + 1)
			choice = max(distributions, key = lambda distribution: mean_samples[distribution])
			histories[agent].append((choice, choice.rvs()))
			if step % skip == 0: plt.plot(tuple(itertools.accumulate([event[1] for event in histories[agent]])), color = 'orange')

	step += 1
	plt.pause(.001)





exit()


dists = [bernoulli(uniform.rvs()) for arm in range(10)]

for dist in dists:
	pass




exit()


arms = range(10)

distributions = {arm: bernoulli(uniform.rvs()) for arm in arms}

agents = {'optimal', 'random', 'greedy'}

histories = {agent: [] for agent in agents}

while True:
	for agent in agents:
		pass










probabilities = scipy.stats.uniform()

arms = range(10)

distributions = {arm: scipy.stats.bernoulli(probabilities.rvs()) for arm in arms}

successes = {arm: 0 for arm in arms}
failures = {arm: 0 for arm in arms}

plt.ion()
plt.show()

earnings = [0]

while True:
	choice = max(arms, key = lambda arm: (successes[arm] + .5) / (successes[arm] + failures[arm] + 1))
	reward = distributions[choice].rvs()
	earnings.append(earnings[-1] + reward)
	plt.plot(earnings, 'r')
	plt.pause(.0001)


