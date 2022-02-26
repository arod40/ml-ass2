from itertools import product
from random import choice, random, seed
import numpy as np

import matplotlib.pyplot as plt


seed(0)


def check_half_circle(circle, upper=True):
    (cx, cy), r = circle
    return lambda x, y: (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2 and (
        upper and y >= cy or not upper and y <= cy
    )


def generate_points(n, circles, box):
    ic1, oc1, ic2, oc2 = circles
    width, height = box

    points = []

    i = 0
    while i < n:
        x = random() * width
        y = random() * height

        label = None
        if oc1(x, y) and not ic1(x, y):
            label = 1

        elif oc2(x, y) and not ic2(x, y):
            label = -1

        if label is not None:
            points.append([[x, y], label])
            i += 1

    return points


def plot_line(w, x1, x2, ax, color="black", label=""):
    C, A, B = w

    y1 = (-C - A * x1) / B
    y2 = (-C - A * x2) / B

    ax.plot([x1, x2], [y1, y2], color=color, label=label)


def plot_data(data, ax):
    X_pos = [point[0] for point, label in data if label == 1]
    Y_pos = [point[1] for point, label in data if label == 1]
    X_neg = [point[0] for point, label in data if label == -1]
    Y_neg = [point[1] for point, label in data if label == -1]
    ax.scatter(X_pos, Y_pos, marker="o", color="blue", label="positive")
    ax.scatter(X_neg, Y_neg, marker="x", color="red", label="negative")


def perceptron(data, max_iter=-1, first_or_choice=True):
    perceptron = lambda w, x: 1 if sum([a * b for a, b in zip(w, x)]) >= 0 else -1
    d = len(data[0][0])
    data = [([1] + x, y) for x, y in data]
    w = [0] * (d + 1)

    it = 0
    while max_iter == -1 or it < max_iter:
        it += 1
        incorrect = [(x, y) for x, y in data if perceptron(w, x) != y]
        if len(incorrect) == 0:
            break
        if it % 1000 == 0:
            print(f"Iteration: {it}, incorrect: {len(incorrect)}")

        if first_or_choice:
            xw, yw = incorrect[0]
        else:
            xw, yw = choice(incorrect)
        w = [a + yw * b for a, b in zip(w, xw)]

    return w, it, len([(x, y) for x, y in data if perceptron(w, x) != y])


def linear_regression_matrix(data):
    X = np.array([[1] + x for x, _ in data])
    y = np.array([y for _, y in data])
    Xt = X.transpose()
    return list(np.matmul(np.matmul(np.linalg.inv(np.matmul(Xt, X)), Xt), y))


rad = 10
thk = 5
sep = 5


box_width = 3 * rad + 2.5 * thk
box_height = 2 * (rad + thk) + sep
centers = [(thk + rad, rad + thk + sep), (1.5 * thk + 2 * rad, rad + thk)]
radius = [rad, rad + thk]
circles = [
    check_half_circle(circle, upper)
    for circle, upper in zip(product(centers, radius), [True, True, False, False])
]

data = generate_points(2000, circles, (box_width, box_height))


fig, ax = plt.subplots()
plot_data(data, ax)

import sys


def linear(w, x):
    w, bias = w[1:], w[0]
    return sum([wi * xi for wi, xi in zip(w, x)]) + bias


def mse(weights):
    return sum([(linear(weights, x) - y) ** 2 for x, y in data]) / len(data)


item = sys.argv[1]
if item == "a":
    w1, it, wrong = perceptron(data, first_or_choice=False)
    plot_line(w1, 0, box_width, ax, label="perceptron", color="black")
    print(mse(w1))


elif item == "b":
    w2 = linear_regression_matrix(data)
    plot_line(w2, 0, box_width, ax, label="linear regression", color="black")
    print(mse(w2))

ax.legend()
plt.show()

