import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 300)
y = 0.5*(np.cos(x/300*np.pi) + 1)*0.05

plt.plot(x, y)
plt.title('Learning Rate Schedule: Cosine Annealing')
plt.xlabel('Training Epoch')
plt.ylabel('Learning Rate')
# plt.grid(True)
plt.show()