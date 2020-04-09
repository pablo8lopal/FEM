#!python3
# FEM1d.py - 1D Linear Finite Element calculation program


# Imports
import numpy as np
import sympy
import math
from numpy.linalg import inv
from scipy.integrate import quad
import matplotlib.pyplot as plt

''' Calculation of the Finite Element Solution of:

    -(p(x)y'(x))' + b(x)y'(x) + q(x)y(x) = f(x)

    where p,b,q and f are to be treated as numbers or symbolically.'''


# Definition of functions
# x=sympy.Symbol('x')
# p = 1 #x**2
# b = 0 #1 / ( x + 1 )
# q = 1 #2 ** x
# f = 1 + 0.5 * x
# Define coefficient values or functions
def p(x):
    return 1.


def b(x):
    return 1.


def q(x):
    return 1.


def f(x):
    return 1. + 0.5 * x


p_v = np.vectorize(p)
b_v = np.vectorize(b)
q_v = np.vectorize(q)
f_v = np.vectorize(f)

# Domain
dom = np.linspace(0, 1, 11)  # Start, End, Nº Points
p_x = p_v(dom)
b_x = b_v(dom)
q_x = q_v(dom)
f_x = f_v(dom)

# plt.plot(dom, p_x)
# plt.plot(dom, b_x)
# plt.plot(dom, q_x)
# plt.plot(dom, f_x)
# plt.show()
# Definition of Boundary Conditions
y0 = 1
y1 = 1

# Definition of Solution Space

nod = dom
nel = np.arange(0, nod.shape[0] - 1)
nnodi = np.arange(0, nel.shape[0])
nnodf = np.arange(1, nod.shape[0])
nodi = nod[nnodi]
nodf = nod[nnodf]
h = nodf - nodi
print(nod)
print(nel)
print(nnodi)
print(nnodf)
print(nodi)
print(nodf)
print(h)

# Definition of Connectivity Table
# TO BE UPDATED IN FUTURE
print('Connectivity Table'.upper().center(70))
print('-' * 70)
print('Element  '
      + '  Initial Node  ' + '  Final Node  '
      + '  Initial Value  ' + '  Final Value')
tcon = np.zeros(shape=(nel.shape[0], 5))
tcon[:, 0] = nel
tcon[:, 1] = nnodi
tcon[:, 2] = nnodf
tcon[:, 3] = nodi
tcon[:, 4] = nodf

print(tcon)

# Definition of Uncoupled Arrays
sdes = np.zeros(shape=(nel.shape[0], 2, 2))
bdes = np.zeros(shape=(nel.shape[0], 2, 2))
mdes = np.zeros(shape=(nel.shape[0], 2, 2))
fdes = np.zeros(shape=(nel.shape[0], 2))
for j in nel:
    # phi1 = -1 / h[j] * (x - nodf[j])
    # phi2 = 1 / h[j] * (x - nodi[j])
    # dphi1 = sympy.diff(phi1, x)
    # dphi2 = sympy.diff(phi2, x)
    def phi1(x):
        return -1 / h[j] * (x - nodf[j])

    def phi2(x):
        return 1 / h[j] * (x - nodi[j])

    def dphi1():
        return -1 / h[j]

    def dphi2():
        return 1 / h[j]

    # Uncoupled S Arrays (y'' term)
    # sdes[j,0,0] = scipy.integrate(p*dphi1*dphi1,
    #                               (x, nodi[j], nodf[j])
    #                               )
    def f1(x):
        return p(x) * dphi1() * dphi1()

    sdes[j, 0, 0] = quad(f1, nodi[j], nodf[j])[0]
    sdes[j, 0, 1] = - sdes[j, 0, 0]
    sdes[j, 1, 0] = sdes[j, 0, 1]  # = - sdes[j,0,0]
    sdes[j, 1, 1] = sdes[j, 0, 0]

    # Uncoupled B Arrays (y' term)
    def f2(x):
        return b(x) * phi1(x) * dphi1()

    def f3(x):
        return b(x) * phi2(x) * dphi1()
    bdes[j, 0, 0] = quad(f2, nodi[j], nodf[j])[0]
    bdes[j, 0, 1] = - bdes[j, 0, 0]
    bdes[j, 1, 0] = quad(f3, nodi[j], nodf[j])[0]
    bdes[j, 1, 1] = - bdes[j, 1, 0]

    # Uncoupled M Arrays (y term)
    def f4(x):
        return q(x) * phi1(x) * phi1(x)

    def f5(x):
        return q(x) * phi1(x) * phi2(x)
    mdes[j, 0, 0] = quad(f4, nodi[j], nodf[j])[0]
    mdes[j, 0, 1] = quad(f5, nodi[j], nodf[j])[0]
    mdes[j, 1, 0] = mdes[j, 0, 1]
    mdes[j, 1, 1] = mdes[j, 0, 0]

    # Uncoupled F Arrays (y term)
    def f6(x):
        return f(x) * phi1(x)

    def f7(x):
        return f(x) * phi2(x)
    fdes[j, 0] = quad(f6, nodi[j], nodf[j])[0]
    fdes[j, 1] = quad(f7, nodi[j], nodf[j])[0]

# Coupling of the matrices S, B, M, F starting from the uncoupled matrices

s = np.zeros(shape=(nod.shape[0], nod.shape[0]))
for i in nel:
    s[i, i] += sdes[i, 0, 0]
    s[i, i+1] = sdes[i, 0, 1]
    s[i+1, i] = sdes[i, 1, 0]
    s[i+1, i+1] = sdes[i, 1, 1]

b = np.zeros(shape=(nod.shape[0], nod.shape[0]))
for i in nel:
    b[i, i] += bdes[i, 0, 0]
    b[i, i+1] = bdes[i, 0, 1]
    b[i+1, i] = bdes[i, 1, 0]
    b[i+1, i+1] = bdes[i, 1, 1]

m = np.zeros(shape=(nod.shape[0], nod.shape[0]))
for i in nel:
    m[i, i] += mdes[i, 0, 0]
    m[i, i+1] = mdes[i, 0, 1]
    m[i+1, i] = mdes[i, 1, 0]
    m[i+1, i+1] = mdes[i, 1, 1]

f = np.zeros(shape=(nod.shape[0]))
for i in nel:
    f[i] += fdes[i, 0]
    f[i+1] = fdes[i, 1]

# Matrix sum and selection of columns and rows

l_mat = s + m + b
L = l_mat[1:nel.shape[0], 1:nel.shape[0]]
F = f[1:nel.shape[0]]
F[0] -= l_mat[1, 0] * y0
F[-1] -= l_mat[-2, -1] * y1
Linv = inv(L)
U = np.dot(Linv, F)


# Add BC
U = np.insert(U, 0, y0, axis=0)
U = np.append(U, y1)
print(U)

plt.plot(dom, U)
plt.show()
