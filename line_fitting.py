import numpy as np
import matplotlib.pyplot as plt

x = np.array([8.9635, 14.7315, 19.2143, 24.4981, 30.9727, 34.8932, 39.4129, 44.1768, 48.9237, 54.3345, 59.4861, 64.1521, 68.9876])
y = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70])
# Reshape x to a column vector
x = x.reshape((-1, 1))

# Create the design matrix A
A = np.hstack([x, np.ones_like(x)])

m, c = np.linalg.lstsq(A, y, rcond=None)[0]

print("Slope (m):", m)
print("Intercept (c):", c)

plt.plot(x, y, 'o', label='Original data', markersize=10)
plt.plot(x, m * x + c, 'r', label='Fitted line')
plt.legend()
plt.show()
