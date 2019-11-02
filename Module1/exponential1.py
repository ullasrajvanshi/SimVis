# CIE4604
# Simulating random variables
"""
Exercises "Inverse-transform method" applied to exponential distribution
     F(x) = 1 - exp(-lambda * x)
Inverse function is  x = Finv(u).
"""
# %% importing libraries
import numpy as np
import matplotlib.pyplot as plt

# %%
# First, generate Nsamp samples from U(0,1)
N_samples = 10000
u = np.random.rand(N_samples, 1)

# Secondly, compute the inverse function x = Finv(u) fron the function u=F(x)
# using pen and paper. Please note that 1-u also has U(0,1) distribution.
# Enter the function here

# %%
l = 0.25  # lambda
x = -(1 / l) * np.log(u)


# %%  plot histograms and calculate mean and standard deviations based on simulated data
def histc(X, bins):
    map_to_bins = np.digitize(X, bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i - 1] += 1
    return [r, map_to_bins]


plt.figure(figsize=(12, 5))
binsize = 2
# plot histogram
Nx = histc(x, np.arange(0, 50, binsize))
plt.bar(np.arange(0, 50, binsize), Nx[0] / (N_samples * binsize))
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Number of samples = %i' %N_samples)

#%% Computing mean and variance
x.mean()
x.var()

