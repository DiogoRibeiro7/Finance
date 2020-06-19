import numpy as np
import scipy.stats
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

T = 1
S_0 = 100;
r = 0.05;

sigma = np.linspace(0.01, 2, num=100)
K = np.linspace(S_0 * 0.1, S_0*2, num=200)

def get_bs_price(s, sigma, t, r, k):
    phi_arg_1 = (np.log(s/k) + (r + 0.5*(sigma**2))* t) / (sigma * np.sqrt(t))
    phi_arg_2 = (np.log(s/k) + (r - 0.5*(sigma**2)) * t) / (sigma * np.sqrt(t))
    phi_1 = scipy.stats.norm.cdf(phi_arg_1)
    phi_2 = scipy.stats.norm.cdf(phi_arg_2)
    price = s * phi_1 - np.exp(-r * t) * k * phi_2
    return price

X,Y = np.meshgrid(sigma, K)
prices = get_bs_price(S_0, X, T, r, Y)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, prices, color='b')
ax.set_xlabel('sigma')
ax.set_ylabel('strike price')
ax.set_zlabel('option price')


n = 10**3
W = np.random.normal(size=n)
outer_prod = np.outer(np.sqrt(T) * sigma, W)
tiled_sigma = np.transpose(np.tile(sigma, (n, 1)))
#S_T has dimensions (length(sigma), n)
S_T = S_0 * np.exp((r - 0.5*(tiled_sigma**2)) * T + outer_prod)

means = np.zeros(prices.shape)
for i,k in enumerate(sorted(K)):
    payoffs = np.clip(S_T - k, a_min=0, a_max=np.Inf)
    means[i,:] = np.exp(-r * T) * np.mean(payoffs, axis=1)

ax.plot_surface(X, Y, means, color='r')
plt.show()
