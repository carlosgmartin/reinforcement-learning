import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

state = [0, 0]

plt.set_cmap('gray')
plt.ion()
plt.show()

size = (4, 6)

axes = plt.axes()

axes.get_xaxis().set_visible(False)
axes.get_yaxis().set_visible(False)
axes.add_patch(Rectangle((size[1] - 1, 0), 1, 1, alpha=.5, fill=True, color='green', linewidth=0))
axes.add_patch(Rectangle((1, 0), 4, 1, alpha=.5, fill=True, color='red', linewidth=0))
axes.set_aspect(1)

rectangle = Rectangle(state, 1, 1, alpha=1, fill=False, color='yellow', linewidth=3)
axes.add_patch(rectangle)

data = np.random.normal(size=size)
mesh = plt.pcolormesh(data)
while True:
	rectangle.set_xy(state)
	data += np.random.normal(size=size, scale=.1)
	mesh.set_array(data.ravel())
	plt.pause(.01)

	state[0] = (state[0] + 1) % size[1]






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






















