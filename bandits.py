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
			greediness = np.exp(-len(histories))
			if scipy.stats.uniform.rvs() > greediness:
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


