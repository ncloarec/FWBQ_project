
from math import exp, log, pi, sqrt, cos
import numpy as np
from copy import copy
from matplotlib import pyplot as plt

# dimensions = d
d = 1

# inputs
# // density p


def p(x):
    sigma = 1
    return(1 / (sqrt(2 * pi)) * exp(-np.dot(x, x.transpose()) / (2 * sigma * sigma)))
# // function f


def f(x):
    return(-np.dot([exp(y) for y in x], x.transpose()))


# --- compute int( f(x)p(x)dx ) using basic Monte Carlo integration
I = []
n = 20
for i in range(1000):
    A = 0
    for j in range(n):
        W = 1 / n
        X = np.random.normal(size=d)
        A += W * f(X)
    I.append(A)
sum(I) / len(I)

# --- compute int( f(x)p(x)dx ) using Frank-Wolfe Bayesian Quadrature

# // number of points = n
n = 20

# // reproducing kernel k


def k(a, b):
    alpha = 1
    sigma = 0.8
    return(alpha * alpha * exp(-np.dot(a - b, (a - b).transpose()) / (2 * sigma * sigma)))
# //


rho = [1 / (i + 1) for i in range(1, n + 1)]


def phi(x):
    def atom(a):
        return(k(a, x))
    return(atom)


def psi(rho, a, b):
    def g(x):
        return((1 - rho) * a(x) + rho * b(x))
    return(g)


dico = {}
i = 1
print("point = ", i)
W = [np.nan]
for l in range(1, i + 1):
    w = 1
    for j in range(l + 1, i + 1):
        w = w * (1 - rho[j - 1]) * rho[l - 1]
    W.append(w)
X = np.random.normal(size=1)
dico[i] = {"point": X, "weight": W}
dico[i]["function"] = phi(x=dico[i]["point"])
g = dico[i]["function"]
# step 1 // FW (Frank-Wolfe) algorithm
# i = 2 to n
for i in range(2, n + 1):
    print("point = ", i)
    # step1 ) computing a new point
    # computing function to minimize

    def T(x, i, g):
        s = 0
        for j in range(1, i):
            w = dico[i - 1]["weight"][j]
            c = dico[j]["point"]
            s = s + (w * k(x, c))
        s = s - g(x)
        return(s)

    # X1 = sorted(np.random.normal(size = 10000))
    # Y1 = [T(x, i=i, g = g) for x in X1]
    # plt.plot(X1, Y1)
    # plt.show()

    # find the minimum
    Xmin = np.random.normal(size=1)
    Tmin = T(x=Xmin, i=i, g=g)
    for _ in range(20000):
        X = np.random.normal(size=1)
        t = T(x=X, i=i, g=g)
        if t < Tmin:
            Xmin = X
            Tmin = t
    X = Xmin
    # step 2 ) computing the weights for the next iteration
    W = [np.nan]
    for l in range(1, i + 1):
        w = 1
        for j in range(l + 1, i + 1):
            w = w * (1 - rho[j - 1]) * rho[l - 1]
        W.append(w)
    dico[i] = {"point": X, "weight": W}
    dico[i]["function"] = phi(x=dico[i]["point"])
    # step 3 ) computing the mean element
    g = psi(rho=rho[i], a=g, b=dico[i]["function"])

R = []
for x in dico.items():
    R.append(float(x[1]["point"]))
    print(x[1]["point"])
print(sorted(R))

# step 2 // BQ (Bayesian Quadrature) algorithm
# compute the vector Z
Z = np.zeros(shape=(n, 1))
for i in range(n):
    Xi = dico[i + 1]["point"]
    print(Xi)
    Z[i][0] = g(Xi)
# compute the matrix K
K = np.zeros(shape=(n, n))
for i in range(n):
    for j in range(n):
        Xi = dico[i + 1]["point"]
        Xj = dico[j + 1]["point"]
        K[i][j] = k(Xi, Xj)
INVK = np.linalg.inv(K)

# weights

WBQ = np.dot(Z.transpose(), INVK)
WBQ
sum(WBQ[0])

# step 3 // posterior mean

F = np.zeros(shape=(n, 1))
for i in range(n):
    Xi = dico[i + 1]["point"]
    F[i][0] = f(Xi)
    print(i, Xi, f(Xi))

mean = np.dot(Z.transpose(), np.dot(INVK, F))

# step 4 // posterior variance

P = np.zeros(shape=(n, 1))
for i in range(n):
    Xi = dico[i + 1]["point"]
    P[i][0] = p(Xi)

# step 5 // full posterior (normal distribution)
# //

# basic monte carlo integration / Frank-Wolfe Bayesian Quadrature integration

print(" monte carlo integration = ", sum(I) / len(I), "\n",
      "Frank-Wolfe Bayesian Quadrature integration =", float(mean))
