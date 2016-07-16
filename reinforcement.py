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




next_state = transition[state][action]
value[state][action] = reward[state][action] + discount * max(value[state][next_action] for next_action in actions[next_state])
state = next_state





action = choose(state, value[state])
next_state = transition(state, action)
value[state][action] = (1 - rate) * value[state][action] + rate * (reward(state, action) + max(value[next_state][next_action] for next_action in actions(state)))
state = next_state






















