def eliminate(r1, r2, col, target=0):
    fac = (r2[col] - target) / r1[col]
    for i in range(len(r2)):
        r2[i] -= fac * r1[i]


def gauss(a):
    for i in range(len(a)):
        if a[i][i] == 0:
            for j in range(i + 1, len(a)):
                if a[i][j] != 0:
                    a[i], a[j] = a[j], a[i]
                    break
            else:
                raise ValueError("Matrix is not invertible")
        for j in range(i + 1, len(a)):
            eliminate(a[i], a[j], i)
    for i in range(len(a) - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            eliminate(a[i], a[j], i)
    for i in range(len(a)):
        eliminate(a[i], a[i], i, target=1)
    return a


def inverse(a):
    tmp = [[] for _ in a]
    for i, row in enumerate(a):
        assert len(row) == len(a)
        tmp[i].extend(row + [0] * i + [1] + [0] * (len(a) - i - 1))
    gauss(tmp)
    ret = []
    for i in range(len(tmp)):
        ret.append(tmp[i][len(tmp[i]) // 2 :])
    return ret


def matmul(A, B):
    assert len(A[0]) == len(B), f"{len(A)}x{len(A[0])} {len(B)}x{len(B[0])}"
    n, shared, m = len(A), len(A[0]), len(B[0])
    C = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for k in range(shared):
                C[i][j] += A[i][k] * B[k][j]
    return C


def transpose(A):
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


n, d = list(map(int, input().split()))
data = []
for _ in range(n):
    line = list(map(float, input().split()))
    data.append((line[:-1], line[-1]))

X = [[1] + x for x, y in data]
y = [[y] for x, y in data]
Xt = transpose(X)
w = [x[0] for x in matmul(matmul(inverse(matmul(Xt, X)), Xt), y)]
print(" ".join([str(x) for x in w]))
