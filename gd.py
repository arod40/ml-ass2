import random
from datetime import datetime

random.seed(0)

time_start = datetime.now()


def gd_step(w, lr, gradient):
    sub_elem_wise = lambda vec1, vec2: [x - y for x, y in zip(vec1, vec2)]
    scalar_mult = lambda scalar, vector: [scalar * x for x in vector]
    return sub_elem_wise(w, scalar_mult(lr, gradient))


def check_bounds(w, lower, upper):
    return all([l <= wi for l, wi in zip(lower, w)]) and all(
        [wi <= u for u, wi in zip(upper, w)]
    )


MAX_QUERIES = 10000
LOW = -1000
UP = 1000

d = int(input())
lower = list(map(float, input().split()))
upper = list(map(float, input().split()))

lr = 10

# Initial value of w chosen at random from the set of valid values
w = [random.random() * (w - l) + l for l, w in zip(lower, upper)]
w_query = w
gradient = [0] * d
last_f = None

it = 0
while True:
    it += 1

    # Shrinking the lr to make sure the query falls within the bounds
    w_query = gd_step(w, lr, gradient)
    while not check_bounds(w_query, lower, upper):
        lr /= 10
        w_query = gd_step(w, lr, gradient)

    # Querying the grader
    print(" ".join([str(wi) for wi in w_query]))
    f = float(input())
    gradient = list(map(float, input().split()))

    # Updating lr and w
    if last_f is None or f < last_f:
        last_f = f
        w = w_query
    # Shrinking the lr in case it has overstepped
    else:
        lr /= 10

    # Checking for breaking conditions
    if it == MAX_QUERIES or (datetime.now() - time_start).seconds > 1.9:
        break

