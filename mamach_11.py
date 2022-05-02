# question 8 verification:
from pprint import pprint

import numpy as np
from sympy import Matrix

import matrix_manipulation
from matrix_manipulation import bmatrix, test_symmetry, matrix_pprint, test_anti_symmetry

A_8 = np.array([
    [1, 1, 1, 1],
    [2, 3, 1, 0],
    [4, 6, -1, 0],
    [0, -2, 5, 4],
    [1, 1, 1, 0],
])

A_8 = Matrix(A_8)

print(A_8.rref())

A_8_b = np.array([
    [-1, 2, 3],
    [2, 1, 1],
    [3, 0, 4],
    [4, -5, 2],
])

print(Matrix(A_8_b).rref())
# print(J)


# question 12 verification:

A_12 = np.array([
    [1, 1, 1, 1],
    [1, 3, 2, 4],
    [2, 0, 1, -1],
])

A_12_t = A_12.transpose()
print(Matrix(A_12_t).rref())
print(Matrix(A_12).rref())

print(Matrix(np.array([[1, 1], [1, -1], [2, 2]])).rref())

# question 13:

a1 = np.array([
    [0, 1],
    [0, 0],
])

a2 = np.array([
    [1, 1],
    [0, 0],
])

print(bmatrix(a2))

# question 14 verification

A_14 = np.array([
    [1, 2],
    [3, 4],
])

B_14 = np.array([
    [5, 6],
    [7, 8],
])

I2 = np.identity(2) * 2
I4 = I2 * 2
# print(I4)
print((A_14 @ A_14) - I4)
print((A_14 - I2) @ (A_14 + I2))

print((A_14 @ A_14) - (B_14 @ B_14))
print((A_14 - B_14) @ (A_14 + B_14))

# A_12_t_copy = A_12_t.copy()
# np.delete(A_12_t_copy, 1)
# print(A_12_t_copy)

print(A_12_t)

a_help = np.array([
    [1, 1, 2],
    [1, 2, 1],
    [1, 4, -1]
])
print(Matrix(A_12_t).rref())

# question 15 verification:

for _ in range(10_000):
    rand_dim = np.random.randint(2, 5)
    A_15 = np.random.rand(rand_dim, rand_dim)
    B_15 = np.random.rand(rand_dim, rand_dim)

    assert test_symmetry(np.array([[0, 1], [1, 2]]))
    assert not test_symmetry(np.array([[0, 1.2], [1, 2]]))

    AB_15 = A_15 @ B_15.transpose() + B_15 @ A_15.transpose()
    assert test_symmetry(AB_15), f"{AB_15} is not symmetric"

    A_15_b = np.random.rand(rand_dim, rand_dim)
    A_15_b = A_15_b + A_15_b.T
    assert test_symmetry(A_15_b)
    B_15_b = np.random.rand(rand_dim, rand_dim)
    B_15_b = B_15_b + -B_15_b.T
    assert test_anti_symmetry(B_15_b)

    mat_pow = np.linalg.matrix_power
    AB_15_b = (mat_pow(A_15_b, 5) @ mat_pow(B_15_b, 3)) - (mat_pow(B_15_b, 3) @ mat_pow(A_15_b, 5))
    assert test_symmetry(AB_15_b), f"{AB_15_b} is not symmetric!"

print("Done! all true")

a = np.array([
    [1, -1],
    [-1, 1],
])
print(matrix_pprint(mat_pow(a, 4)))

# question 16 verification:
for _ in range(100):
    singular = np.array([
        [1, 0],
        [1, 0]
    ])
    # print(np.linalg.inv(singular))
    # print(Matrix(singular).rref())
    assert np.linalg.matrix_rank(singular) < singular.shape[0]


    def random_op():
        return np.random.choice(list(matrix_manipulation.ElementaryOperation))


    operations = [(np.random.randint(1, 3), np.random.randint(1, 3), np.random.randint(-3, 3), random_op())
                  for _ in range(10)]

    messed_up_singular = matrix_manipulation.do_series_of_operations(matrix=singular, operations=operations).astype(
        float)
    # print(matrix_pprint(messed_up_singular.astype(int)))
    assert np.linalg.matrix_rank(messed_up_singular) < messed_up_singular.shape[0]

print("singular assumptions hold!")


# question 20

C_20 = np.array([
    [1, 0, 0],
    [2, 1, 0],
    [-14, -7, 0],
])

A_20 = np.array([
    [1, -2, 4],
    [0, 5, -13],
    [0, 0, 0],
])

print(bmatrix(C_20))

# question 21 verification:

A_22 = np.array([
    [1, 0, -2, 1],
    [2, 1, 0, 2],
    [-1, 1, -2, 1],
    [3, 1, -1, 0],
])

print(Matrix(A_22).det())


# question 9 verification:
A_9_a = np.array([
    [1, 1, 1],
    [0, 1, 2],
    [3, 0, 1],
])
print(Matrix(A_9_a).rref())

print(Matrix(np.array([[1,2,3],[4,5,6],[7,8,9]])).det())


no_span = np.array([[1, 0, 0], [0, 1, 0]])
r3_span = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
r3_ld = np.array([[1, 0, 0], [0, 1, 0], [0, 2, 0]])
print(bmatrix(r3_ld))
print(Matrix(r3_ld).rref())