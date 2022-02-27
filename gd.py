from datetime import datetime
from random import random
from math import sqrt, inf


time_start = datetime.now()


def check_bounds(w, lower=None, upper=None):
    lower = lower or [-1] * len(w)
    upper = upper or [1] * len(w)
    return all([l <= wi for l, wi in zip(lower, w)]) and all(
        [wi <= u for u, wi in zip(upper, w)]
    )


MAX_QUERIES = 10000
LOW = -1000
UP = 1000

d = int(input())
lower = list(map(float, input().split()))
upper = list(map(float, input().split()))

lr = 0.1
eps = 1e-8
beta = 0.9
f_prev = inf
w_prev = None

w = [random() * (u - l) + l for u, l in zip(upper, lower)]
gradient = [0] * d
v = [0] * d

elem_wise = lambda vec1, vec2, op: [op(x, y) for x, y in zip(vec1, vec2)]
scalar_mult = lambda scalar, vector: [scalar * x for x in vector]
add = lambda x, y: x + y
mult = lambda x, y: x * y
sub = lambda x, y: x - y
div = lambda x, y: x / y

it = 0
while it < MAX_QUERIES:  # and (datetime.now() - time_start).seconds < 1.9:
    it += 1

    # Querying the grader
    print(" ".join([str(wi) for wi in w]))
    f = float(input())
    gradient = list(map(float, input().split()))

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
        w = [random() * (u - l) + l for u, l in zip(upper, lower)]
        f_prev = inf
        w_prev = None
        lr = 0.1
        v = [0] * d
    else:
        f_prev = f
        w_prev = w
