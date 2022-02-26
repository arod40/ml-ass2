from random import random, seed
import matplotlib.pyplot as plt
import numpy as np


seed(0)


def get_gD(x1, x2):
    return lambda x: (x1 + x2) * x - x1 * x2


def get_random_points(M=1000):
    return [random() * 2 - 1 for _ in range(M)]


def get_random_pairs(N=100):
    return list(zip(get_random_points(N), get_random_points(N)))


print("Computing gbar...")
N = 100
gbar_pairs = get_random_pairs(N)
gbar = lambda x: sum([get_gD(x1, x2)(x) for x1, x2 in gbar_pairs]) / N

N = 1000
M = 10000
pairs = get_random_pairs(N)
points = get_random_points(M)
gbar = lambda x: sum([get_gD(x1, x2)(x) for x1, x2 in pairs]) / N
gbar_memo = {x: gbar(x) for x in points}


print("Computing bias...")
bias = sum([(gbar_memo[x] - x ** 2) ** 2 for x in points]) / M


print("Computing variance...")
var = sum(
    [(get_gD(x1, x2)(x) - gbar_memo[x]) ** 2 for x in points for x1, x2 in pairs]
) / (N * M)

print("Computing Eout...")
eout = sum([(get_gD(x1, x2)(x) - x ** 2) ** 2 for x in points for x1, x2 in pairs]) / (
    N * M
)


print(eout, bias, var, bias + var)


def plot_function(f, ax, bounds, label, color="black"):
    X = np.linspace(start=bounds[0], stop=bounds[1], num=100)
    y = np.vectorize(f)(X)

    ax.plot(X, y, label=label, color=color)


f = lambda x: x ** 2
gbar = lambda x: 0

fig, ax = plt.subplots()
plot_function(f, ax, [-1, 1], "f", color="blue")
plot_function(gbar, ax, [-1, 1], "gbar", color="red")
plt.legend()
plt.show()
