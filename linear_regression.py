from datetime import datetime
from random import random
from math import sqrt, inf

time_start = datetime.now()
_inf = 2e9

elem_wise = lambda vec1, vec2, op: [op(x, y) for x, y in zip(vec1, vec2)]
scalar_mult = lambda scalar, vector: [scalar * x for x in vector]
add = lambda x, y: x + y
mult = lambda x, y: x * y
sub = lambda x, y: x - y
div = lambda x, y: x / y


def gradient_descent(target, derivative, bounds, max_steps=10000, max_time=2):
    def check_bounds(w, lower=None, upper=None):
        lower = lower or [-1] * len(w)
        upper = upper or [1] * len(w)
        return all([l <= wi for l, wi in zip(lower, w)]) and all(
            [wi <= u for u, wi in zip(upper, w)]
        )

    best_f = inf
    best_w = None

    lower, upper = bounds
    d = len(lower)

    lr = 0.1
    eps = 1e-8
    beta = 0.9
    f_prev = inf
    w_prev = None

    w = [0 for u, l in zip(upper, lower)]
    gradient = [0] * d
    v = [0] * d

    it = 0
    while it < max_steps and (datetime.now() - time_start).seconds < max_time:
        it += 1

        f = target(w)
        gradient = derivative(w)

        if f < best_f:
            best_w = w
            best_f = f

        # Shrink lr if overstepped
        if f > f_prev and it > 10:
            lr /= 10
            w = w_prev

        # Updating v
        v = [
            sqrt(x) + eps
            for x in elem_wise(
                scalar_mult(beta, v),
                scalar_mult(1 - beta, elem_wise(gradient, gradient, mult)),
                add,
            )
        ]
        checked = False
        # Adjusting learning rate if stepped out of the bounds
        for i in range(10):
            w = elem_wise(w, scalar_mult(lr, elem_wise(gradient, v, div)), sub)

            if check_bounds(w, lower, upper):
                checked = True
                break
            lr /= 10

        # If never got within bounds or change is less than epsilon, start over from another point
        if not checked or abs(f_prev - f) < eps:
            w = [0 for u, l in zip(upper, lower)]
            f_prev = inf
            w_prev = None
            lr = 0.1
            v = [0] * d
        else:
            f_prev = f
            w_prev = w

    return best_w, best_f, it, lr


def normalize_data(data):
    norm_data = data
    weight_norm_inv = lambda x: x
    return norm_data, weight_norm_inv


def linear_regression(data):
    data, weight_norm_inv = normalize_data(data)

    N = len(data)
    d = len(data[0][0])

    def linear(w, x):
        w, bias = w[1:], w[0]
        return sum([wi * xi for wi, xi in zip(w, x)]) + bias

    def mse(weights):
        return sum([(linear(weights, x) - y) ** 2 for x, y in data]) / N

    def mse_derivative(weights):
        gradient = [0] * (d + 1)

        for x, y in data:
            value = [linear(weights, x) - y] + scalar_mult(linear(weights, x) - y, x)
            gradient = elem_wise(gradient, value, add)

        return [2 * g / N for g in gradient]

    w, _, _, _ = gradient_descent(
        mse,
        mse_derivative,
        bounds=([-_inf] * (d + 1), [_inf] * (d + 1)),
        max_steps=100000,
        max_time=2,
    )

    w = weight_norm_inv(w)

    return w


n, d = list(map(int, input().split()))
data = []
for _ in range(n):
    line = list(map(float, input().split()))
    data.append((line[:-1], line[-1]))

print(" ".join([str(x) for x in linear_regression(data)]))
