# %% Multivariate distribution
#
# In this example a multivariate distribution |z| with expectation |mu| and co-variance matrix |Sigma| (Qyy) is simulated.
# We use the Choleski decomposition to factor the co-variance matrix,  such that |Sigma = R*R'|, with |R| the Choleski factor.

# %% importing libraries
import numpy as np
import matplotlib.pyplot as plt

# %%
mu = np.array([1, 2])
sigma = np.array([[1, 0.5], [0.5, 2]])

#%%
# Cholesky factor
R = np.linalg.cholesky(sigma)
# Generate samples (why numpy.kron?)
z = np.kron(np.ones((10000,1)),mu) + np.dot(np.random.randn(10000,2),R)

#%%
plt.figure(figsize=(6,6))
plt.plot(z[:,0], z[:,1],'.')
plt.axis('equal')
plt.xlabel(r'$z_1$')
plt.ylabel(r'$z_2$')
plt.title('Multivariate')

