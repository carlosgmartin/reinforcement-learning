import numpy as np
import matplotlib.pyplot as plt

plt.ion()
plt.show()

history = [0]
history2 = [0]

while True:
	history.append(history[-1] + np.random.choice([1, -1]))
	history2.append(history2[-1] + np.random.choice([2, -2]))
	plt.plot(history, 'b')
	plt.plot(history2, 'g')
	plt.pause(.0001)