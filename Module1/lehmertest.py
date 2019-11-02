# %% import the required libraries
import numpy as np
import matplotlib.pyplot as plt

# %%
xrand = np.random.rand(1000)
plt.figure(figsize=(12, 8))
plt.psd(xrand)
plt.title('Periodogram Power Spectral Density (rand)')
plt.tight_layout()
plt.show()


# %%
def lehmer(n=None, x0=None, a=None, m=None):
    """
    :param n: numbers
    :param x0: using X0 as starting value
    :param a: multiplier A
    :param m: modulus M
    :return: uniformly distributed random numbers

    Examples:

        x=lehmer(1000,1,13,31);        %  bad choice
        x=lehmer(1000,1,7^5,2^31-1);   %  older Matlab versions

    (c) Hans van der Marel, 2019

    Created: 1 Novermber 2019 by Hans van der Marel
    """
    x = np.zeros((n))
    x[0] = x0
    for k in range(1, n):
        x[k] = np.mod(a * x[k - 1], m)
    x = x / m
    return x


# %%
x = lehmer(1000, 1, 13, 31)
plt.figure(figsize=(12, 8))
plt.psd(x)
plt.title('Periodogram Power Spectral Density (Legacy Matlab)')

# %%
plt.figure(figsize=(12, 8))
plt.plot(x[0:29], x[30:59], '*')
plt.title('2D scatter (60 samples)')
plt.xlabel('x(0:29)')
plt.ylabel('x(30:59)')
# %%
plt.figure(figsize=(12, 8))
plt.plot(x[0:499], x[500:999], '*')
plt.title('2D scatter (1000 samples)')
plt.xlabel('x(0:499)')
plt.ylabel('x(500:999)')
