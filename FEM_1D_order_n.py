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


# Define coefficient values or functions
def p(x):
    return 1. + 10*x**2


def b(x):
    return 20*math.cos(x)


def q(x):
    return 1.


def f(x):
    return 1. + 0.5 * x


# Specify boundary conditions
# At left end
y0 = 0
# At right end
y1 = 1


# Define order of FE Solution
order = 1

# Define domain and boundary conditions
# Estimate number of points for calculation
num_points = 1001
if order > 1:
    if num_points % order != 1:
        num_points = math.ceil(num_points / order) * order + 1

# Specify Start and End Point of Domaijn
domain = np.linspace(0, 10, num_points)  # Start, End, Nº Points

# Element Matrices
num_elements = (num_points - 1) // order
nel = np.arange(0, num_elements)

# Node Vectors
nod_vec = domain
nnod_vec = np.arange(0, nod_vec.shape[0])

# Global nodes by element
nod_global = np.zeros((num_elements, order+1), dtype=np.float)
nnod_global = np.zeros((num_elements, order+1), dtype=np.int8)
for elem in nel:
    nod_global[elem, :] = nod_vec[elem*order:(elem+1)*order+1]
    nnod_global[elem, :] = nnod_vec[elem*order:(elem+1)*order+1]

# Subelement width by element
h = np.zeros((num_elements, order), dtype=np.float)
for col in range(order):
    h[:, col] = nod_global[:, col + 1] - nod_global[:, col]

print(nod_global)
print(nnod_global)
print(h)


# Initialize Uncoupled Matrices
sdes = np.zeros(shape=(nel.shape[0], order + 1, order + 1))
bdes = np.zeros(shape=(nel.shape[0], order + 1, order + 1))
mdes = np.zeros(shape=(nel.shape[0], order + 1, order + 1))
fdes = np.zeros(shape=(nel.shape[0], order + 1))

for elem in nel:

    # Generate Shape Functions
    phi = []
    dphi = []
    nod_elem = nod_global[elem, :]
    for n in range(order + 1):

        def phi_func(x, n=n):
            func = 1
            for point in range(order + 1):
                if point != n:
                    func *= (x - nod_elem[point]) / (nod_elem[n] - nod_elem[point])
            return func
        phi.append(phi_func)

        def dphi_func(x, n=n):
            func = 0
            for i in range(order + 1):
                if i != n:
                    tmp = 1
                    for m in range(order + 1):
                        if (m != n) and (m != i):
                            tmp *= (x - nod_elem[m]) / (nod_elem[n] - nod_elem[m])
                    func += ((1 / (nod_elem[n] - nod_elem[i])) * tmp)
            return func
        dphi.append(dphi_func)

    # SDES MATRICES -> SYMMETRIC
    for i in range(order+1):
        for j in range(order+1):
            if i > j:
                sdes[elem, i, j] = sdes[elem, j, i]
            else:
                # Define temp function
                def temp(x, i=i, j=j):
                    return p(x) * dphi[i](x) * dphi[j](x)
                sdes[elem, i, j] = quad(temp, nod_elem[0], nod_elem[-1])[0]

    # BDES MATRICES -> NOT SYMMETRIC. TODO: CHECK!!
    for i in range(order+1):
        for j in range(order+1):
            # Define temp function
            def temp(x, i=i, j=j):
                return b(x) * phi[i](x) * dphi[j](x)
            bdes[elem, i, j] = quad(temp, nod_elem[0], nod_elem[-1])[0]

    # MDES MATRICES -> SYMMETRIC
    for i in range(order+1):
        for j in range(order+1):
            if i > j:
                mdes[elem, i, j] = mdes[elem, j, i]
            else:
                # Define temp function
                def temp(x, i=i, j=j):
                    return q(x) * phi[i](x) * phi[j](x)
                mdes[elem, i, j] = quad(temp, nod_elem[0], nod_elem[-1])[0]

    # FDES MATRICES
    for i in range(order+1):
        # Define temp function
        def temp(x, i=i):
            return f(x) * phi[i](x)
        fdes[elem, i] = quad(temp, nod_elem[0], nod_elem[-1])[0]


# Coupling of the matrices S, B, M, F starting from the uncoupled matrices

s = np.zeros(shape=(nod_vec.shape[0], nod_vec.shape[0]))
b = np.zeros(shape=(nod_vec.shape[0], nod_vec.shape[0]))
m = np.zeros(shape=(nod_vec.shape[0], nod_vec.shape[0]))
f = np.zeros(shape=(nod_vec.shape[0]))

for elem in nel:
    s[elem*order:(elem+1)*order+1, elem*order:(elem+1)*order+1] += sdes[elem, :, :]
    b[elem*order:(elem+1)*order+1, elem*order:(elem+1)*order+1] += bdes[elem, :, :]
    m[elem*order:(elem+1)*order+1, elem*order:(elem+1)*order+1] += mdes[elem, :, :]
    f[elem*order:(elem+1)*order+1] += fdes[elem, :]


# Matrix sum and selection of columns and rows
# TODO: Fix so it works for order "n"
l_mat = s + m + b
L = l_mat[1:nod_vec.shape[0] - 1, 1:nod_vec.shape[0] - 1]
F = f[1:nod_vec.shape[0] - 1]
F[:] -= l_mat[1:nod_vec.shape[0] - 1, 0] * y0
F[:] -= l_mat[1:nod_vec.shape[0] - 1, -1] * y1
Linv = inv(L)
U = np.dot(Linv, F)


# Add BC
U = np.insert(U, 0, y0, axis=0)
U = np.append(U, y1)
print(U)

plt.plot(domain, U)
plt.show()
