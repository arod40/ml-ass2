import random
from datetime import datetime

random.seed(0)

time_start = datetime.now()


def update_lr(it, f_history, lr):
    return lr


def check_bounds(w, lower, upper):
    return all([l <= wi for l, wi in zip(lower, w)]) and all(
        [wi <= u for u, wi in zip(upper, w)]
    )


def gd_step(w, lr, gradient):
    sub_elem_wise = lambda vec1, vec2: [x - y for x, y in zip(vec1, vec2)]
    scalar_mult = lambda scalar, vector: [scalar * x for x in vector]
    return sub_elem_wise(w, scalar_mult(lr, gradient))


MAX_QUERIES = 10000
LOW = -1000
UP = 1000

d = int(input())
lower = list(map(float, input().split()))
upper = list(map(float, input().split()))

lr = 0.1

# Initial value of w chosen at random from the set of valid values
w = [random.random() * (w - l) + l for l, w in zip(lower, upper)]
history = []

it = 0
while True:
    it += 1

    # Querying the grader
    print(" ".join([str(wi) for wi in w]))
    f = float(input())
    gradient = list(map(float, input().split()))

    # GD step
    w = gd_step(w, lr, gradient)

    # Updating history and lr
    history.append((f, gradient))
    lr = update_lr(it, history, lr)

    # Checking for breaking conditions
    if (
        it == MAX_QUERIES
        or not check_bounds(w, lower, upper)
        or (datetime.now() - time_start).seconds > 1.9
    ):
        break

