import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import collections

# Size of the world
size = (6, 4)
tiles = {(x, y): 'empty' for x in range(size[0]) for y in range(size[1])}
tiles[size[0] - 1, 0] = 'goal'
for x in range(1, size[0] - 1):
	tiles[x, 0] = 'cliff'

# Agent that chooses randomly between actions
class RandomAgent:
	def observe(self, state):
		pass

	def choose(self, actions):
		return random.choice(actions)

	def receive(self, reward):
		pass



# Agent that values an action according to its immediate reward
# and chooses an action using a softmax selection rule
class ImmediateRewardAgent:
	# Returns the softmax (normalized exponential) of a vector
	def softmax(self, vector, temperature = .1):
		factors = np.exp(np.array(vector) / temperature)
		total = np.sum(factors)
		probabilities = factors / total
		return probabilities

	def __init__(self):
		self.values = collections.defaultdict(lambda: 0)

	def observe(self, state):
		self.state = state # Remember current state of the environment

	def choose(self, actions):
		# Choose an action according to a softmax action selection rule
		values = [self.values[self.state, action] for action in actions]
		action = np.random.choice(actions, p=self.softmax(values))
		self.action = action # Remember action that was performed
		return action

	def receive(self, reward):
		self.values[self.state, self.action] = reward



# Agent that performs Q-learning
class QAgent:
	# Returns the softmax (normalized exponential) of a vector
	def softmax(self, vector, temperature = 10):
		factors = np.exp(np.array(vector) / temperature)
		total = np.sum(factors)
		probabilities = factors / total
		return probabilities

	def __init__(self):
		self.states = []
		self.actions = []
		self.rewards = []
		self.values = collections.defaultdict(lambda: 0)

	def observe(self, state):
		self.states.append(state)

	def choose(self, actions):
		current_state = self.states[-1]
		self.values[current_state] = max(self.values[current_state, action] for action in actions)
		
		# Choose an action according to a softmax action selection rule
		values = [self.values[current_state, action] for action in actions]
		action = np.random.choice(actions, p=self.softmax(values))
		# action = random.choice(actions)
		self.actions.append(action)

		if len(self.rewards) > 0:
			last_state = self.states[-2]
			last_action = self.actions[-2]
			reward = self.rewards[-1]

			discount = .99
			rate = 1 # For a deterministic environment
			occurrences = len([pair for pair in zip(self.states, self.actions) if pair == (last_state, last_action)])
			rate = 1 / occurrences

			if False:
				# SARSA (on-policy) learning: use value under action that was actually taken
				self.values[last_state, last_action] = (1 - rate) * self.values[last_state, last_action] + rate * (reward + discount * self.values[current_state, action])
			else:
				# Q (off-policy) learning: use value under best action
				self.values[last_state, last_action] = (1 - rate) * self.values[last_state, last_action] + rate * (reward + discount * self.values[current_state])

		return action

	def receive(self, reward):
		self.rewards.append(reward)



# Agent that performs Q-learning
class QAgentBroken:
	# Returns the softmax (normalized exponential) of a vector
	def softmax(self, vector, temperature = .01):
		factors = np.exp(np.array(vector) / temperature)
		total = np.sum(factors)
		probabilities = factors / total
		return probabilities

	def __init__(self):
		self.values = collections.defaultdict(lambda: 0)

	def observe(self, state):
		try:
			self.last_state = self.state # Remember last state of the environment
		except AttributeError:
			print('First timestep')
		self.state = state # Remember current state of the environment

	def choose(self, actions):
		# Choose an action according to a softmax action selection rule
		values = [self.values[self.state, action] for action in actions]
		action = np.random.choice(actions, p=self.softmax(values))

		try:
			self.values[self.state] = max(self.values[self.state, action] for action in actions)
			self.last_action = self.action # Remember the last action that was performed
			discount = .99
			rate = 1
			self.values[last_state, last_action] = (1 - rate) * self.values[last_state, last_action] + rate * (self.reward + discount * self.values[state])
		except (AttributeError, NameError):
			print('First timestep')

		self.action = action # Remember the current action
		return action

	def receive(self, reward):
		self.reward = reward # Remember reward that was received



actions = ['left', 'right', 'up', 'down']

def rewards(state, action):
	next_state = transition(state, action)
	if tiles[next_state] == 'goal': # If the agent moves into the goal
		return 1
	elif tiles[next_state] == 'cliff':
		return -100
	else:
		return 0

def transition(state, action):
	if tiles[state] in ['cliff', 'goal']:
		return (0, 0)

	if action == 'left' and state[0] > 0:
		return (state[0] - 1, state[1])
	elif action == 'right' and state[0] < size[0] - 1:
		return (state[0] + 1, state[1])
	elif action == 'down' and state[1] > 0:
		return (state[0], state[1] - 1)
	elif action == 'up' and state[1] < size[1] - 1:
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

mesh = plt.pcolormesh(np.zeros(size).transpose(), vmin=-1, vmax=1)

goal_rect = matplotlib.patches.Rectangle((size[0] - 1, 0), 1, 1, alpha=.5, fill=True, color='green', linewidth=0)
cliff_rect = matplotlib.patches.Rectangle((1, 0), size[0] - 2, 1, alpha=.5, fill=True, color='red', linewidth=0)
agent_rect = matplotlib.patches.Rectangle(state, 1, 1, alpha=1, fill=False, color='yellow', linewidth=3)

plt.axes().add_patch(goal_rect)
plt.axes().add_patch(cliff_rect)
plt.axes().add_patch(agent_rect)

while True:
	print('\nState: {}'.format(state))
	agent.observe(state)
	action = agent.choose(actions)
	print('Action: {}'.format(action))
	reward = rewards(state, action)
	print('Reward: {}'.format(reward))
	agent.receive(reward)
	state = transition(state, action)

	# Draw state (position) of agent in the world
	agent_rect.set_xy(state)

	# If agent is keeping track of values for each state in the world
	# then display these values overlayed on the world
	if hasattr(agent, 'values'):
		array = np.array([[agent.values[x, y] for y in range(size[1])] for x in range(size[0])])
		mesh.set_array(array.transpose().ravel())

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
	plt.pause(.01)

	action = agent.choose(actions)
	print(action)
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






















