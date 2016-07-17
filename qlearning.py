import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import collections

# Size of the world
size = (6, 4)

# Agent that chooses randomly between actions
class RandomAgent:
	def __init__(self):
		self.state_values = np.zeros(size)
	def observe(self, state):
		pass
	def choose(self, actions):
		return random.choice(actions)
	def receive(self, reward):
		pass

# Agent that performs Q-learning
class QAgent:
	# Returns the softmax, or normalized exponential, of a vector
	def softmax(self, vector):
		factors = np.exp(np.array(vector))
		total = np.sum(factors)
		probabilities = factors / total
		return probabilities
	def __init__(self):
		# Values assigned to states
		self.state_values = np.zeros(size)
		# Values assigned to state-action pairs
		self.values = collections.defaultdict(lambda: 0)
		# Number of occurrences for each state-action pair
		self.counts = collections.defaultdict(lambda: 0)
	def observe(self, state):
		self.state = state # Remember current state of the environment when choosing action
	def choose(self, actions):
		# Choose an action according to a softmax action selection rule
		values = [self.values[self.state, action] for action in actions]
		return np.random.choice(actions, p=self.softmax(values))
	def receive(self, reward):
		

		pass

actions = ['left', 'right', 'up', 'down']

def rewards(state, action):
	return 0

def transition(state, action):
	if action == 'left' and state[0] > 0:
		return (state[0] - 1, state[1])
	elif action == 'right' and state[0] < size[0] - 1:
		return (state[0] + 1, state[1])
	elif action == 'up' and state[1] > 0:
		return (state[0], state[1] - 1)
	elif action == 'down' and state[1] < size[1] - 1:
		return (state[0], state[1] + 1)
	else:
		return state

agent = QAgent()
state = (0, 0)

plt.set_cmap('gray')
plt.ion()
plt.show()
plt.axes().set_aspect(1)
plt.axes().get_xaxis().set_visible(False)
plt.axes().get_yaxis().set_visible(False)

mesh = plt.pcolormesh(np.empty(size).transpose(), vmin=-1, vmax=1)

goal_rect = matplotlib.patches.Rectangle((2, 2), 1, 1, alpha=.5, fill=True, color='green', linewidth=3)
agent_rect = matplotlib.patches.Rectangle(state, 1, 1, alpha=1, fill=False, color='yellow', linewidth=3)

plt.axes().add_patch(goal_rect)
plt.axes().add_patch(agent_rect)

while True:
	print '\nState: {}'.format(state)
	agent.observe(state)
	action = agent.choose(actions)
	print 'Action: {}'.format(action)
	reward = rewards(state, action)
	print 'Reward: {}'.format(reward)
	agent.receive(reward)
	state = transition(state, action)

	agent_rect.set_xy(state)
	mesh.set_array(agent.state_values.transpose().ravel())

	plt.pause(.01)








exit()






import random

# Agent that chooses randomly between actions
class RandomAgent:
	def observe(self, state):
		pass
	def choose(self, actions):
		return random.choice(actions)
	def receive(self, reward):
		pass

agent = RandomAgent()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def transition(state, action):
	next_state = np.array(state) + np.array(action)
	if next_state[0] < 0:
		next_state[0] = 0
	elif next_state[0] > size[0] - 1:
		next_state[0] = size[0] - 1
	if next_state[1] < 0:
		next_state[1] = 0
	elif next_state[1] > size[1] - 1:
		next_state[1] = size[1] - 1
	return next_state


state = [0, 0]

plt.set_cmap('gray')
plt.ion()
plt.show()

size = (6, 4)

axes = plt.axes()

axes.get_xaxis().set_visible(False)
axes.get_yaxis().set_visible(False)
axes.add_patch(Rectangle((size[0] - 1, 0), 1, 1, alpha=.5, fill=True, color='green', linewidth=0))
axes.add_patch(Rectangle((1, 0), 4, 1, alpha=.5, fill=True, color='red', linewidth=0))
axes.set_aspect(1)

rectangle = Rectangle(state, 1, 1, alpha=1, fill=False, color='yellow', linewidth=3)
axes.add_patch(rectangle)

data = np.random.normal(size=size)
mesh = plt.pcolormesh(data.transpose())

actions = [[1, 0], [0, 1], [-1, 0], [0, -1]]

while True:
	rectangle.set_xy(state)
	data += np.random.normal(size=size, scale=.02)
	mesh.set_array(data.transpose().ravel())
	plt.pause(.1)

	action = agent.choose(actions)
	print action
	state = transition(state, action)









exit()










import math
import random
import np

def softmax(vector):
	factors = map(math.exp, vector)
	total = sum(factors)
	return [factor / total for factor in factors]

def choose(values, state):
	action_values = values[state]
	probabilities = softmax(action_values)
	return np.random.choice(actions, probabilities)

actions = range(5)
states = range(5)

rewards = [
	[-1, -1, -1, -1, 0, -1],
	[-1, -1, -1, 0, -1, 100],
	[-1, -1, -1, 0, -1, -1],
	[-1, 0, 0, -1, 0, -1],
	[0, -1, -1, 0, -1, 100],
	[-1, 0, -1, -1, 0, 100]
]

q = [[0] * 5] * 5

while True:
	state = random.choice(states)
	while state != 5:
		pass



next_state = transition[state][action]
value[state][action] = reward[state][action] + discount * max(value[state][next_action] for next_action in actions[next_state])
state = next_state





action = choose(state, value[state])
next_state = transition(state, action)
value[state][action] = (1 - rate) * value[state][action] + rate * (reward(state, action) + max(value[next_state][next_action] for next_action in actions(state)))
state = next_state






















