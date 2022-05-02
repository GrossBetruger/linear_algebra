from fractions import Fraction
from functools import reduce

import numpy as np
import pandas as pd

from matrix_manipulation import do_row_operation, ElementaryOperation, do_series_of_operations, bmatrix, matrix_pprint

add = ElementaryOperation.Add

A = np.array([
    [1, 0, 2, 3],
    [2, -1, 3, 6],
    [1, 4, 4, 0],
])


A1 = do_row_operation(A, left_row=2, right_row=3, factor=-2, op=add)
print(A1.astype("int64"))
print()
# [[ 1  0  2  3]
#  [ 0 -9 -5  6]
#  [ 1  4  4  0]]


A2 = do_row_operation(A1, left_row=3, right_row=1, factor=-1, op=add)
print(A2.astype("int64"))
print()
# [[ 1  0  2  3]
#  [ 0 -9 -5  6]
#  [ 0  4  2 -3]]

A3 = do_row_operation(A2, left_row=3, right_row=2, factor=Fraction(4, 9), op=add)
print(pd.DataFrame(A3))
print()

# 1   0     2     3
# 0  -9    -5     6
# 0   0  -2/9  -1/3

A4 = do_row_operation(A3, left_row=1, right_row=3, factor=9, op=add)
print(pd.DataFrame(A4))
print()
#
# 1   0     0     0
# 0  -9    -5     6
# 0   0  -2/9  -1/3


A5 = do_row_operation(matrix=A4, left_row=3, factor=Fraction(-9, 2), op=ElementaryOperation.Factor)
print(pd.DataFrame(A5))
print()
#
# 1   0   0    0
# 0  -9  -5    6
# 0   0   1  3/2

A6 = do_row_operation(matrix=A5, left_row=2, right_row=3, factor=5, op=add)
print(pd.DataFrame(A6))
print()
#
# 1   0  0     0
# 0  -9  0  27/2
# 0   0  1   3/2

A7 = do_row_operation(matrix=A6, left_row=2, factor=Fraction(-1, 9), op=ElementaryOperation.Factor)
print(pd.DataFrame(A7))
print()
#
# 1  0  0     0
# 0  1  0  -3/2
# 0  0  1   3/2


operations = [
    (2, 3, -2, add),
    (3, 1, -1, add),
    (3, 2, Fraction(4, 9), add),
    (1, 3, 9, add),
    (3, None, Fraction(-9, 2), ElementaryOperation.Factor),
    (2, 3, 5, add),
    (2, None, Fraction(-1, 9), ElementaryOperation.Factor),
]

res = do_series_of_operations(A, operations)
print(pd.DataFrame(res))

I_3 = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
])

P = do_series_of_operations(I_3, operations)
print(pd.DataFrame(P))

R = P @ A
print()
print(pd.DataFrame(R))
print(np.array_equal(R, do_series_of_operations(A, operations=operations)))

print()
print("P=")
print(bmatrix(P.astype(str)).replace("'", ''))
print()


latex_I_matrices = []
phi_matrices = []
for i, op in enumerate(reversed(operations)):
    left, right, factor, op = op
    phi_I = do_row_operation(matrix=I_3, left_row=left, right_row=right, factor=factor, op=op)
    phi_matrices.append(phi_I)
    print(f"phi: {7-i}")
    print(matrix_pprint(phi_I))
    latex_mat = bmatrix(phi_I.astype(str)).replace("'", '')
    latex_I_matrices.append(latex_mat)
    print()

print()
print()

print("\\times".join(latex_I_matrices))

mult = reduce(lambda a, b: a@b, phi_matrices)
print(matrix_pprint(mult))

print()
print(matrix_pprint(mult @ A))


# question 4

A = np.array([
    [0, 1, 0, 0, 0],
    [0, 0, 2, 0, 0],
    [0, 0, 0, 3, 0],
    [0, 0, 0, 0, 4],
    [5, 0, 0, 0, 0],
])

print()
print(np.linalg.inv(A) @ A)\


# question 7

A = np.array([
    [0, 1, 1],
    [-1, 0, 1],
    [-1, -1, 0],
])

# assert A antisymmetric:

assert np.array_equal(A.transpose(), -A)

B = np.array([
    [0, 1, 2],
    [1, 0, 3],
    [2, 3, 0],
])

# B symmetric:

assert np.array_equal(B, B.transpose())

# (A+B)^2
ab_2 = (A+B) @ (A+B)

assert not np.array_equal(ab_2, ab_2.transpose())

print(bmatrix(A))
print(
)
print(bmatrix(B))

print()
print(bmatrix(ab_2.transpose()))


An_5 = np.array([
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 2],
    [1, 1, 1, 3, 1],
    [1, 1, 4, 1, 1],
    [1, 5, 1, 1, 1],
])

An_6 = np.array([
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 2],
    [1, 1, 1, 1, 3, 1],
    [1, 1, 1, 4, 1, 1],
    [1, 1, 5, 1, 1, 1],
    [1, 6, 1, 1, 1, 1],
])

An_7 = np.array([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 2],
    [1, 1, 1, 1, 1, 3, 1],
    [1, 1, 1, 1, 4, 1, 1],
    [1, 1, 1, 5, 1, 1, 1],
    [1, 1, 6, 1, 1, 1, 1],
    [1, 7, 1, 1, 1, 1, 1],
])

An_4 = np.array([
    [1, 1, 1, 1],
    [1, 1, 1, 2],
    [1, 1, 3, 1],
    [1, 4, 1, 1],
])

An_3 = np.array([
    [1, 1, 1],
    [1, 1, 2],
    [1, 3, 1],
])

An_5_tag = np.array([
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 2, 0],
    [0, 0, 3, 0, 0],
    [0, 4, 0, 0, 0],
])

print(np.linalg.det(An_5))
print(np.linalg.det(An_6))
print(np.linalg.det(An_7))
print(np.linalg.det(An_4))
print(np.linalg.det(An_3))
print(np.linalg.det(An_5_tag))
print(np.linalg.det(An_7))
# print()
# print(np.math.factorial(4))
# print(np.math.factorial(5))
# print(np.math.factorial(6))
# print(np.math.factorial(3))


def det_sign(num_row_swaps: int, order: int) -> int:
    sign = 1 if order % 2 == 0 else -1
    if (num_row_swaps - 1) % 2 == 1:
        sign *= -1
    return sign


print([det_sign(n-1, n) for n in range(1, 10)])
print()
print()

for n in range(1, 9):
    mat = np.zeros((n, n))
    mat[0] = mat[0] + 1
    for i in range(n-1):
        mat[i+1][-1-i] = i + 1
    print(mat)
    print(np.linalg.det(mat))
    print()