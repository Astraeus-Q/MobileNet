import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Set the parameter α
alpha = 0.2

# Generate a range of values for x
x = np.linspace(0, 1, 1000)

# Compute the PDF of the Beta distribution
pdf = beta.pdf(x, alpha, alpha)

# Plot the PDF
plt.plot(x, pdf, label=f'α={alpha}')
plt.title('PDF of Beta Distribution, α = 0.2')
plt.xlabel('x')
plt.ylabel('Probability Density')
# plt.legend()
plt.show()